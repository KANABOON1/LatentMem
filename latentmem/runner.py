import logging
import os
from dataclasses import asdict
from typing import Literal

import torch
from common.config import Config
from common.interactions import (
    InteractionConfig, 
    InteractionDataProto,                             
    lazy_get_inter_cls
)
from common.utils.tensor_utils import (
    fix_model_parameters,                                       
    log_trainable_params
)
from data.base_builder import BaseDataBuilder
from data.base_env import BaseEnv
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig
from trl import GRPOConfig, SFTConfig

from latentmem.mas_core.base_memory_mas import BaseMemoryMAS
from latentmem.trainer import (
    MemMasterLMPOTrainer,
    MemMasterSFTTrainer
)
from latentmem.utils.data import (
    rag_to_structured_json,
    structured_json_to_trajectories
)
from latentmem.utils.message import Trajectory
from latentmem.utils.tools import remove_trainer_ckpts


class LatentMemRunner:

    def __init__(
        self, memory_mas: BaseMemoryMAS, data_builder: BaseDataBuilder, config: Config, working_dir: str
    ):
        # memory mas
        self.memory_mas = memory_mas
        self.processing_class = self.memory_mas.shared_tokenizer

        # dataset
        dataset_dict: DatasetDict = data_builder.get_dataset_dict()
        self.train_dataset = dataset_dict["train"]
        self.valid_dataset = dataset_dict["valid"]
        self.test_dataset = dataset_dict["test"]

        self.config = config
        self.working_dir = working_dir
        self.env_cls = data_builder.get_env_cls()

        # parse generation config
        self.generation_config = GenerationConfig(**self.config.run_cfg.generation)
        self.generation_config.bos_token_id = memory_mas.agents_list[0].tokenizer.bos_token_id
        self.generation_config.pad_token_id = memory_mas.agents_list[0].tokenizer.pad_token_id
        self.generation_config.eos_token_id = memory_mas.agents_list[0].tokenizer.eos_token_id

        # get interaction controller 
        self.interaction_config = InteractionConfig(**self.config.run_cfg.interaction)
        interaction_class = lazy_get_inter_cls(self.env_cls)
        self.interaction_controller = interaction_class(self.memory_mas, self.interaction_config, self.generation_config)

        # parse training config
        updated_inputs = {
            "output_dir": working_dir,
            "report_to": ["tensorboard", "wandb"]
        }
        sft_inputs = dict(**self.config.run_cfg.sft, **updated_inputs)
        self.sft_config = SFTConfig(**sft_inputs)
        grpo_inputs = dict(**self.config.run_cfg.grpo, **updated_inputs)
        self.grpo_config = GRPOConfig(**grpo_inputs)
    
    
    def _set_batch_envs(self, batch) -> tuple[list[str], list[str], list[BaseEnv]]:
        
        system_prompts, init_user_prompts, envs = [], [], []
        
        for task_config in batch:
            env = self.env_cls(self.config.dataset_cfg)
            system_prompt, init_user_prompt = env.set_env(task_config)

            system_prompts.append(system_prompt)
            init_user_prompts.append(init_user_prompt)
            envs.append(env)
        
        return system_prompts, init_user_prompts, envs

    def bootstrap_data(self):
        device = torch.device("cpu" if self.config.run_cfg.device == -1 else f"cuda:{self.config.run_cfg.device}")
        self.memory_mas.to(device)
        fix_model_parameters(self.memory_mas)
        log_trainable_params(self.memory_mas)

        batch_size = self.interaction_config.batch_size
        bootstrap_data_dataloader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: batch
        )

        total_completed = 0
        total_correct = 0

        logging.info(f"Total size of data: {len(bootstrap_data_dataloader)}")
        for batch_id, bootstrap_batch in tqdm(enumerate(bootstrap_data_dataloader)):

            data_batch = InteractionDataProto()
            domain_instructions, task_descriptions, envs = self._set_batch_envs(bootstrap_batch)
            data_batch.no_tensor_batch['domain_instructions'] = domain_instructions
            data_batch.no_tensor_batch['task_descriptions'] = task_descriptions
            data_batch.no_tensor_batch['envs'] = envs

            gen_batch = self.interaction_controller.run_inter_loop(data_batch)
            trajectories: list[Trajectory] = gen_batch.no_tensor_batch["trajectories"]
            
            output_path = os.path.join(self.working_dir, "trajectories_output.txt")
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"=== Trajectory (batch {batch_id}) ===\n")
                for trajectory in trajectories:
                    if trajectory.label is not None:
                        total_completed += 1
                        if trajectory.label == True:
                            total_correct += 1
                            self.memory_mas.centralized_memory.add_memory(trajectory)
            
                    text_repr = trajectory.to_text()
                    f.write(text_repr)
                    f.write("\n" + "-" * 40 + "\n")
                    
            avg_accuracy = (total_correct / total_completed) if total_completed > 0 else 0.0
            logging.info(f"Average accuracy: {avg_accuracy:.4f} "
                        f"({total_correct}/{total_completed})")
        
        rag_dir = os.path.join(self.working_dir, "rag_0") 
        rag_to_structured_json(rag_dir)

    def _get_sft_dataset(self, json_file_path: str) -> tuple[Dataset, Dataset]:
        
        trajectories = structured_json_to_trajectories(json_file_path)  # list[Trajectory]
        
        message_nodes = []
        for trajectory in trajectories:
            for message_graph in trajectory.trajectory:
                for n, data in message_graph.mas_message_graph.nodes(data=True):
                    message_nodes.append(data["message"])

        # train test split
        train_messages, valid_messages = train_test_split(message_nodes, test_size=0.2, random_state=42)
        
        train_dataset = Dataset.from_list([asdict(msg) for msg in train_messages])
        valid_dataset = Dataset.from_list([asdict(msg) for msg in valid_messages])

        return train_dataset, valid_dataset

    def _get_grpo_dataset(self, mode: Literal["in_mas"]) -> tuple[Dataset, Dataset]:
        rate = 1
        if mode == "in_mas":
            rate = 4
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if self.config.dataset_cfg.name == "kodcode":
            train_size = len(self.train_dataset) // rate
        elif self.config.dataset_cfg.name == "triviaqa":
            train_size = len(self.train_dataset) // rate
        elif self.config.dataset_cfg.name == "popqa":
            train_size = len(self.train_dataset) // rate
        elif self.config.dataset_cfg.name == "strategyqa":
            train_size = len(self.train_dataset)
        else:
            raise ValueError("No such dataset.")
        train_indices = range(train_size)

        valid_size = 1
        valid_indices = range(valid_size)

        return self.train_dataset.select(train_indices), self.valid_dataset.select(valid_indices)

    def train(self, mode: Literal["sft", "grpo"]):
        
        log_trainable_params(self.memory_mas)

        if mode == "sft":
            bootstrapped_data_file_path = self.config.dataset_cfg.bootstrapped_data_path
            train_dataset, valid_dataset = self._get_sft_dataset(bootstrapped_data_file_path)
            
            # initialize SFTTrainer
            self.sft_config.save_strategy = "no"
            self.trainer = MemMasterSFTTrainer(
                self.memory_mas,
                args=self.sft_config,
                data_collator=None,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
            )
        
        elif mode == "lmpo":
            self.grpo_config.do_eval = False
            self.grpo_config.eval_strategy = 'no'
            self.grpo_config.save_strategy = 'no'
            train_dataset, valid_dataset = self._get_grpo_dataset(mode="in_mas")

            # reset lmpo generation_manager
            self.interaction_controller.generation_config.do_sample = True
            self.interaction_controller.generation_config.temperature = self.grpo_config.temperature
            self.interaction_controller.generation_config.max_new_tokens = self.grpo_config.max_completion_length

            # initialize GRPOTrainer
            self.trainer = MemMasterLMPOTrainer(
                model=self.memory_mas,
                reward_funcs=[self.env_cls.compute_reward],
                args=self.grpo_config,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                processing_class=self.processing_class,
                # interaction with env
                env_class=self.env_cls,
                env_main_config=self.config.dataset_cfg,
                generation_manager=self.interaction_controller
            )            
       
        self.trainer.train()
        self.trainer.save_model()  # save the best model
        remove_trainer_ckpts(self.trainer.args.output_dir)  # remove trainer checkpoints

    def evaluate(self):
        device = torch.device("cpu" if self.config.run_cfg.device == -1 else f"cuda:{self.config.run_cfg.device}")
        self.memory_mas.to(device)
        fix_model_parameters(self.memory_mas)
        log_trainable_params(self.memory_mas)
                
        batch_size = self.interaction_config.batch_size
        bootstrap_data_dataloader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: batch
        )

        total_completed = 0
        total_correct = 0
        total_reward = 0

        for batch_id, bootstrap_batch in tqdm(enumerate(bootstrap_data_dataloader)):

            data_batch = InteractionDataProto()
            domain_instructions, task_descriptions, envs = self._set_batch_envs(bootstrap_batch)
            data_batch.no_tensor_batch['domain_instructions'] = domain_instructions
            data_batch.no_tensor_batch['task_descriptions'] = task_descriptions
            data_batch.no_tensor_batch['envs'] = envs

            gen_batch = self.interaction_controller.run_inter_loop(data_batch)
            trajectories: list[Trajectory] = gen_batch.no_tensor_batch["trajectories"]

            output_path = os.path.join(self.working_dir, "trajectories_output.txt")
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"=== Trajectory (batch {batch_id}) ===\n")
                for trajectory in trajectories:
                    if trajectory.label is not None:
                        total_completed += 1
                        total_reward += trajectory.reward
                        if trajectory.label == True:
                            total_correct += 1
                            self.memory_mas.centralized_memory.add_memory(trajectory)
            
                    text_repr = trajectory.to_text()
                    f.write(text_repr)
                    f.write("\n" + "-" * 40 + "\n")

            avg_accuracy = (total_correct / total_completed) if total_completed > 0 else 0.0
            logging.info(f"Average accuracy: {avg_accuracy:.4f} "
                        f"({total_correct}/{total_completed})")
            
            avg_reward = (total_reward / total_completed) if total_completed > 0 else 0.0
            logging.info(f"Average reward: {avg_reward:.4f} "
                        f"({total_reward}/{total_completed})")
            
    def execute(self, mode: str):

        if mode == "data":
            return self.bootstrap_data()
        elif mode == "sft" or mode == "lmpo":
            return self.train(mode)
        elif mode == "eval":
            return self.evaluate()
        else:
            raise ValueError(f"Unsupported mode: {mode}")
