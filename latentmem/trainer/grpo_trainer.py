import copy
import inspect
import networkx as nx
import os
import random
import re
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sequence, Sized
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available
from trl import GRPOTrainer
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import selective_log_softmax
from trl.extras.profiling import profiling_context, profiling_decorator

from common.interactions import InteractionManager, InteractionDataProto, lazy_get_inter_cls
from latentmem.mas_core.base_memory_mas import BaseMemoryMAS
from latentmem.utils.message import MessageNode, Trajectory
from latentmem.trainer.utils import (
    nanmin, nanmax, nanstd
)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class MemMasterLMPOTrainer(GRPOTrainer):
    
    def __init__(
        self,
        model: BaseMemoryMAS,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # --- add interaction between agent and env into grpo trainer ---
        env_class = None,   # env main class
        env_main_config = None,  # configs to initialize an env object
        generation_manager: InteractionManager = None  # manage the interaction between agent and env
    ):  

        args.gradient_accumulation_steps = len(model.agents_list)

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config
        )

        self.env_class = env_class
        self.env_main_config = env_main_config
        self.generation_manager = generation_manager

    def _build_multiturn_envs(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]) -> tuple[list, list, list]:
        domain_instructions, task_descriptions, envs = [], [], []

        for task_config in inputs:
            env = self.env_class(self.env_main_config)
            system_prompt, init_user_prompt = env.set_env(task_config)
            
            domain_instructions.append(system_prompt)
            task_descriptions.append(init_user_prompt)
            envs.append(env)
        
        return domain_instructions, task_descriptions, envs
    
    def _flatten_trajectory(self, trajectories: list[Trajectory]) -> dict[str, list[MessageNode]]:

        agent_messages: dict[str, list[MessageNode]] = dict()
        for trajectory in trajectories:
            if not trajectory.trajectory:
                raise ValueError("Trajectory should not be empty.")
            
            message_graph = trajectory.trajectory[0].mas_message_graph  
            
            for node_id, node_data in message_graph.nodes(data=True):
                message_node: MessageNode = node_data.get('message')

                if node_id not in agent_messages:
                    agent_messages[node_id] = []
                
                agent_messages[node_id].append(message_node)

        message_lengths = {role: len(messages) for role, messages in agent_messages.items()}
        lengths = list(message_lengths.values())
        min_length = min(lengths)
        max_length = max(lengths)       

        if min_length != max_length:
            error_details = {role: length for role, length in message_lengths.items() if length != max_length}
            raise ValueError(
                f"❌ Agent message counts are not uniform across all roles. "
                f"Expected {max_length}, found inconsistent counts for roles: {error_details}"
            )

        return agent_messages

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            agents_num = len(self.model.agents_list)  
            assert self.args.gradient_accumulation_steps == agents_num, "The gradient accumulation steps should match the number of agents in the MAS."
            # num_iterations: how many times this rollout batch is used to update the policy
            # steps_per_generation: how many mini-batches this rollout batch is split into per iteration
            # generate_every = self.args.steps_per_generation * self.num_iterations
            generate_every = self.args.gradient_accumulation_steps
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = [
                    dict(
                        messages=messages, 
                        advantages=generation_batch["advantages"]
                    ) 
                    for messages in generation_batch["messages"].values()
                ]

            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]  
            self._step += 1
        else:  
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch 
            generation_batch = self._generate_and_score_completions(generation_batch)          
            mas_messages = []
            for agents_messages in generation_batch["messages"].values():
                mas_messages.extend(agents_messages)

            agents_num = len(self.model.agents_list)
            advantages = generation_batch["advantages"].repeat(agents_num)
            inputs = dict(messages=mas_messages, advantages=advantages)

        return inputs
    
    def _get_per_token_logps(
        self, 
        model: BaseMemoryMAS, 
        messages: list[MessageNode], 
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        
        outputs = model.forward(inputs=messages, mode='agent')
        # shift logits 和 labels
        logits = outputs.logits[:, :-1]
        labels = outputs["labels"][:, 1:]

        completion_ids = torch.where(labels==-100, torch.zeros_like(labels), labels)
        logps = selective_log_softmax(logits, completion_ids)
        logps[labels == -100] = 0.0

        return logps, labels
        

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]  # batch_size * num_generations
    ) -> dict[str, Union[torch.Tensor, Any]]:
        
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        domain_instructions, task_descriptions, envs = self._build_multiturn_envs(inputs)
        
        gen_batch = InteractionDataProto()
        gen_batch.no_tensor_batch["domain_instructions"] = domain_instructions
        gen_batch.no_tensor_batch["task_descriptions"] = task_descriptions
        gen_batch.no_tensor_batch["envs"] = envs
        
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                # Use GenerationManager to coordinate the interaction between the agent and the environment
                self.generation_manager.memory_mas = unwrapped_model  
                final_gen_batch_output = self.generation_manager.run_inter_loop(gen_batch=gen_batch)
        
        trajectories: list[Trajectory] = final_gen_batch_output.no_tensor_batch["trajectories"]

        rewards_per_func = [traj.label for traj in trajectories] 
        rewards = torch.tensor(rewards_per_func, dtype=torch.float32, device=device).view(-1) 

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # --- log ---
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        
        mas_messages = self._flatten_trajectory(trajectories)
        
        output = {
            "messages": mas_messages,  
            "advantages": advantages,
        }
        
        return output

    def _compute_loss(self, model, inputs):
        device = self.accelerator.device
        
        # messages
        agent_messages = inputs["messages"]
        
        # compute new policy
        per_token_logps, per_token_labels = self._get_per_token_logps(model, agent_messages)
        completion_mask = (per_token_labels != -100).long()

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # loss functions  
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
        mode = "train" if self.model.training else "eval"
        
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count
                    
        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss