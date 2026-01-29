from abc import ABC, abstractmethod
import logging

import torch
import torch.nn as nn
from transformers import GenerationConfig, AutoTokenizer

from typing import Optional, Literal, Union

from common.registry import registry
from common.utils.tensor_utils import load_state_dict_from_safetensor
from memmaster.mas_core.base_centralized_memory import BaseCentralizedMemory
from memmaster.utils.message import MessageNode, MessageGraph
from memmaster.utils.agent import LLMAgent
from memmaster.utils.tools import CONVERSATION_TEMPLATE

class BaseMemoryMAS(ABC, nn.Module): 
    """Memory-based Multi-Agent System
    """
    def __init__(
        self, 
        llm_name_or_path: str, 
        centralized_memory: Optional[BaseCentralizedMemory] = None,
        share_llm: bool = True,
        task_domain: Optional[str] = None,
        **kwargs
    ):  
        super().__init__()

        # mas centralized memory: connecting all agents
        self.centralized_memory = centralized_memory

        # all agents in mas share the tokenizer
        self.llm_name_or_path = llm_name_or_path
        self.shared_tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)

        if self.shared_tokenizer.chat_template is None:
            logging.info("The model does not have a chat template; replace the model's chat template.")
            self.shared_tokenizer.chat_template = CONVERSATION_TEMPLATE
        if self.shared_tokenizer.pad_token_id is None or self.shared_tokenizer.pad_token is None:
            logging.info("The model does not have a pad token, so the eos token is used as the pad token instead.")
            self.shared_tokenizer.pad_token = self.shared_tokenizer.eos_token
            self.shared_tokenizer.pad_token_id = self.shared_tokenizer.eos_token_id

        self.share_llm = share_llm
        self.task_domain = task_domain
        self.agents_list: list[LLMAgent] = list()  
        
        self.warnings_issued = {}
        self.model_tags = None

    def add_model_tags(self, tags: Union[list[str], str]) -> None:
        r"""
        Add custom tags into the model that gets pushed to the Hugging Face Hub. Will
        not overwrite existing tags in the model.

        Args:
            tags (`Union[list[str], str]`):
                The desired tags to inject in the model

        Examples:

        ```python
        from transformers import AutoModel

        model = AutoModel.from_pretrained("google-bert/bert-base-cased")

        model.add_model_tags(["custom", "custom-bert"])

        # Push the model to your namespace with the name "my-custom-bert".
        model.push_to_hub("my-custom-bert")
        ```
        """
        if isinstance(tags, str):
            tags = [tags]

        if self.model_tags is None:
            self.model_tags = []

        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)        
        
    def _agent_forward(self, inputs: list[MessageNode]) -> dict:
        system_prompts, user_prompts, responses = [], [], []
        system_prompt_fields: list[dict] = []
        user_prompt_fields: list[dict] = []
        states: list[dict] = []

        for message_node in inputs:
            # reconstruct system prompt
            system_prompts.append(message_node.system_prompt_template or "")
            system_prompt_fields.append(message_node.system_prompt_fields)
            
            # reconstruct user prompt
            user_prompts.append(message_node.user_prompt_template or "")
            user_prompt_fields.append(message_node.user_prompt_fields)
            
            # label
            responses.append(message_node.response or "")
            
            # reconstruct agent's info
            states.append(message_node.state)
        
        if len(self.agents_list) == 0:
            raise ValueError()
        
        # --- agent forward ---
        outputs = self.agents_list[0].forward(
            system_prompt_templates=system_prompts, 
            system_prompt_fields=system_prompt_fields,
            user_prompt_templates=user_prompts,
            user_prompt_fields=user_prompt_fields,
            responses=responses,  # label
            states=states
        )   

        return outputs
        
    def forward(self, inputs: tuple[list[MessageNode], list[MessageGraph]], mode: Literal["agent"], **kwargs) -> dict:
        if mode == "agent":
            return self._agent_forward(inputs)
        else:
            raise ValueError("Unsupported Mode")

    @abstractmethod
    def generate(self, task_domain_instructions: list[str], user_inputs: list[str], generation_config: GenerationConfig) -> list[MessageGraph]:
        ...

    @classmethod
    def from_config(cls, config, working_dir: str, task_domain: str):

        mas_cfg = config.get("mas", dict())
        memory_cfg = config.get("memory", dict())
        
        memory_name = memory_cfg.get("name")
        memory_cls = registry.get_memory_class(memory_name)
        centralized_memory = memory_cls.from_config(memory_cfg, working_dir)
        
        memory_mas = cls(centralized_memory=centralized_memory, task_domain=task_domain , **mas_cfg)

        load_model_path: str = config.get("load_model_path", None)
        if load_model_path is not None:
            if load_model_path.endswith("safetensors"):
                model_state_dict = load_state_dict_from_safetensor(load_model_path)
                memory_mas.load_state_dict(model_state_dict, strict=False)
            elif load_model_path.endswith("bin"):
                model_state_dict = torch.load(load_model_path)
                memory_mas.centralized_memory.load_state_dict(model_state_dict)
            logging.info(f"Load model state dict from: {load_model_path}")
        
        memory_mas.to(dtype=torch.bfloat16)

        return memory_mas

    def to(self, device=None, dtype=None, *args, **kwargs) -> "BaseMemoryMAS":
        for agent in self.agents_list:
            if device is not None:
                agent.model.to(device=device)
            if dtype is not None:
                agent.model.to(dtype=dtype)
        
        if device is not None:
            self.centralized_memory = self.centralized_memory.to(device=device)
            self.centralized_memory.rag_memory.llm.to(device=device)
            self.centralized_memory.rag_memory.embedding_function.embed_model.to(device=device)
        if dtype is not None:
            self.centralized_memory = self.centralized_memory.to(dtype=dtype)
            self.centralized_memory.rag_memory.llm.to(dtype=dtype)
            self.centralized_memory.rag_memory.embedding_function.embed_model.to(dtype=dtype)        

        return self
        