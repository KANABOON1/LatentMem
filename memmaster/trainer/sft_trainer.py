from collections import defaultdict
from typing import Any, Optional, Union

from datasets import Dataset, IterableDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollator,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import (
    is_peft_available,
    is_sagemaker_mp_enabled
)
from trl.trainer.sft_config import SFTConfig

from memmaster.mas_core.base_memory_mas import BaseMemoryMAS
from memmaster.utils.message import MessageNode

if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat

class MemMasterSFTTrainer(Trainer):

    def __init__(
        self,
        model: BaseMemoryMAS,  
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
    ):
    
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=model.shared_tokenizer,
            compute_loss_func=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=None,
        )
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  
            num_workers=self.args.dataloader_num_workers,
        )
    
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  
            num_workers=self.args.dataloader_num_workers,
        )
    
    def compute_loss(self, model, inputs: list[dict[str, str]], return_outputs=False, num_items_in_batch=None):
        
        mode = "train" if self.model.training else "eval"
        
        message_nodes = [MessageNode(**input) for input in inputs]
        outputs = model.forward(message_nodes, mode="agent")
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss        
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        # return_loss = inputs.get("return_loss")
        # if return_loss is None:
        #     return_loss = self.can_return_loss
        # NOTE - modified
        return_loss = True  
        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.detach().mean()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs

                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)