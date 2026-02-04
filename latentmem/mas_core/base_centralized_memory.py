from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Optional, Union

import torch

@dataclass
class Memory:  # self-evolving memory
    text_memory: Optional[str] = None
    latent_memory: Optional[torch.FloatTensor] = None
    extra_fields: dict = field(default_factory=dict)

class BaseCentralizedMemory(ABC, torch.nn.Module):  
    
    def __init__(self):
        super().__init__()
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

    def register_agents(self, agents_list: list["LLMAgent"]):
        logging.info(f"Register {len(agents_list)} agents into the memory.")
    
    @abstractmethod
    def add_memory(self, **kwargs):
        ...

    @abstractmethod
    def retrieve_memory(self, agent_uuid: str) -> Memory:
        ...

    @abstractmethod
    def process_memory(self, text_memory: str, **kwargs) -> Memory:
        ...
    
    @abstractmethod
    def from_config(self, config, working_dir):
        ...