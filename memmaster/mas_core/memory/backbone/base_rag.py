from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import os
import shutil
from typing import Optional

from accelerate import Accelerator
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memmaster.utils.agent import LLMAgent
from memmaster.utils.message import Trajectory
from common.utils.tensor_utils import fix_model_parameters

from memmaster.utils.tools import CONVERSATION_TEMPLATE

from .utils import EmbeddingFunction

@dataclass
class RAGMemoryResponse:
    pos_shots: list[Trajectory] = field(default_factory=list)
    neg_shots: list[Trajectory] = field(default_factory=list)
    extra_fields: dict = field(default_factory=dict)

class BaseRAGMemory(ABC):

    def __init__(
        self, 
        pos_shots_num: int,
        neg_shots_num: int,
        embedding_model_name_or_path: str,
        llm_name_or_path: str,
        working_dir: str,
        database_dir: Optional[str],
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        self.pos_shots_num = pos_shots_num
        self.neg_shots_num = neg_shots_num
        self.embedding_function = EmbeddingFunction(embedding_model_name_or_path, device=device)
        
        self.working_dir = f"{working_dir}_0"
        
        if database_dir is not None:
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
            try:
                shutil.copytree(database_dir, self.working_dir)
                logging.info(f"The original database has been copied from {database_dir} to the process-private directory: {self.working_dir}.")
            except Exception as e:
                logging.error(f"Database copy failed: {e}")
                raise e
        else:
            logging.info(f"No initial database provided; will run in a newly created directory: {self.working_dir}")
        
        self.main_memory = Chroma(          
            embedding_function=self.embedding_function,      # caculate the vector(embedding)
            persist_directory=self.working_dir,           # store directory
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = CONVERSATION_TEMPLATE
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
        fix_model_parameters(self.llm)
        fix_model_parameters(self.embedding_function.embed_model)

    @abstractmethod
    def add(self, mas_trajectory: Trajectory):
        ...

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> RAGMemoryResponse:
        ...