import os
from typing import Optional

from peft import PeftConfig, LoraConfig
import torch
import torch.nn as nn

from common.registry import registry
from latentmem.mas_core.base_centralized_memory import BaseCentralizedMemory
from latentmem.mas_core.base_centralized_memory import Memory
from latentmem.mas_core.memory.backbone import get_rag_cls, RAGMemoryResponse
from latentmem.mas_core.memory.weaver import Weaver
from latentmem.mas_core.memory import prompt
from latentmem.utils.agent import LLMAgent
from latentmem.utils.message import Trajectory

@registry.register_memory("latentmem")
class LatentMem(BaseCentralizedMemory):
    def __init__(
        self,
        llm_name_or_path: str,
        # rag part
        rag_mode: str,
        pos_shots_num: int,
        neg_shots_num: int,
        insights_num: int,
        embedding_model_name_or_path: str,
        rag_working_dir: str,
        database_dir: Optional[str],

        # weaver part
        use_weaver: bool,
        latents_len: int,
        peft_config: PeftConfig,

    ):  
        super().__init__()

        # rag module
        rag_memory_cls = get_rag_cls(rag_mode)
        self.rag_memory = rag_memory_cls(
            pos_shots_num=pos_shots_num,
            neg_shots_num=neg_shots_num,
            insights_num=insights_num,
            embedding_model_name_or_path=embedding_model_name_or_path,
            working_dir=rag_working_dir,
            database_dir=database_dir,  
            llm_name_or_path=llm_name_or_path
        )

        # weaver module
        self.use_weaver = use_weaver

        if self.use_weaver:
            self.weaver = Weaver(
                llm_name_or_path,
                latents_len=latents_len,
                peft_config=peft_config,
            )
            self.config = self.weaver.config
        else:
            self.weaver = None
            self.config = dict()
       
    def register_agents(self, agents_list: list[LLMAgent]):
        super().register_agents(agents_list)
        
        if self.use_weaver:
            agent_hidden_size = agents_list[0].model.config.hidden_size
            self.weaver2agent_proj = nn.Linear(self.config.hidden_size, agent_hidden_size)
        

    def add_memory(self, mas_trajectory: Trajectory):
        self.rag_memory.add(mas_trajectory)

    
    def retrieve_memory(self, task_description: str, agent: LLMAgent) -> Memory:
        memory = Memory()
        
        retrieved_memory: RAGMemoryResponse = self.rag_memory.retrieve(task_description, agent_role=agent.role)
        
        memory.text_memory = self._construct_text_memory(retrieved_memory)
        
        agent_responses = self._concat_agents_response(retrieved_memory, agent.role)
        if self.use_weaver:
            memory.latent_memory = self._construct_latent_memory(task_description, memory.text_memory, agent.role, agent_responses)

        memory.extra_fields = retrieved_memory.extra_fields
        memory.extra_fields["agent_responses"] = agent_responses
        
        return memory
    
    def process_memory(self, text_memory: str, task_description: str, extra_fields: dict, agent: LLMAgent) -> Memory:
        memory = Memory(text_memory=text_memory)  
        agent_responses = extra_fields.get("agent_responses")
        memory.latent_memory = self._construct_latent_memory(task_description, text_memory, agent.role, agent_responses)

        return memory
    
    def _concat_agents_response(self, retrieved_memory: RAGMemoryResponse, agent_role: str) -> str:

        agent_responses = ""  

        pos_shots = retrieved_memory.pos_shots
        if pos_shots is not None and len(pos_shots) > 0: 
            agent_responses = '\n'.join([pos_shot.extra_fields.get(agent_role, "") for pos_shot in pos_shots]).strip()
        
        return agent_responses
    
    def _construct_text_memory(self, retrieved_memory: RAGMemoryResponse) -> str:
        pos_shots = retrieved_memory.pos_shots
        neg_shots = retrieved_memory.neg_shots
        insights = retrieved_memory.extra_fields.get("insights") if retrieved_memory.extra_fields is not None else None
        
        memory_modules = []
        
        if insights is not None and len(insights) > 0:  # insights
            insights_text = '\n'.join(insights)
            insights_text_memory = prompt.INSIGHTS_TEMPLATE.format(insights=insights_text)
            memory_modules.append(insights_text_memory)
        
        if pos_shots is not None and len(pos_shots) > 0:  # positive trajectory and abstract
            if pos_shots[0].extra_fields.get("abstract") is not None:
                pos_shots_text_memory = None
            else:
                pos_shots_text = '\n'.join([pos_shot.to_text() for pos_shot in pos_shots])
                pos_shots_text_memory = prompt.POS_SHOTS_TEMPLATE.format(memory_few_shots=pos_shots_text)
            memory_modules.append(pos_shots_text_memory)
        
        if neg_shots is not None and len(neg_shots) > 0:  # negtive trajectory and abstract
            if neg_shots[0].extra_fields.get("abstract") is not None:
                neg_shots_text_memory = None
            else:
                neg_shots_text = '\n'.join([neg_shot.to_text() for neg_shot in neg_shots])
                neg_shots_text_memory = prompt.NEG_SHOTS_TEMPLATE.format(memory_few_shots=neg_shots_text)
            memory_modules.append(neg_shots_text_memory)
        
        text_memory = '\n'.join(memory_modules)
        if text_memory == "":  
            text_memory = None  
        return text_memory
        
    def _construct_latent_memory(self, task_description: str, text_memory: str, agent_role: str, agent_responses: str) -> torch.FloatTensor:
        if agent_responses == "" or agent_responses == None:
            extract_latent_prompt = prompt.EXTRACT_LATENT_PROMPT.format(
                current_task=task_description, 
                text_memory=text_memory,
                role=agent_role, 
                k=self.weaver.config.latents_len
            )
        else:
            extract_latent_prompt = prompt.EXTRACT_LATENT_PROMPT_FULL.format(
                current_task=task_description, 
                text_memory=text_memory,
                role=agent_role, 
                agent_message=agent_responses, 
                k=self.weaver.config.latents_len
            )
        latent_memory_embeds = self.weaver.text_to_latent(extract_latent_prompt).squeeze(0)  
        
        latent_memory_embeds = self.weaver2agent_proj(latent_memory_embeds)

        return latent_memory_embeds  

    @classmethod
    def from_config(cls, config, working_dir: str):  
         
        # --- general configs ---
        llm_name_or_path = config.get("llm_name_or_path")

        # --- rag parameters ---
        rag_config = config.get("rag", dict())

        rag_mode = rag_config.get("mode", "gmemory")
        pos_shots_num = rag_config.get("pos_shots_num", 1)
        neg_shots_num = rag_config.get("neg_shots_num", 0)
        insights_num = rag_config.get("insights_num", 3)
        database_dir = rag_config.get("database_dir", None)
        embedding_model_name_or_path = rag_config.get("embedding_model_name_or_path", None)
        if embedding_model_name_or_path is None:
            raise ValueError("`embedding_model_name_or_path` cannot be None.")

        if working_dir is None:
            raise ValueError("`working_dir` cannot be None.")

        # --- weaver parameters ---
        use_weaver = config.get("use_weaver", True)
        weaver_config = config.get("weaver", dict())
        
        latents_len = weaver_config.get("latents_len", 8)
        
        use_peft = weaver_config.get("use_peft", True)
        peft_config = weaver_config.get("peft_config", None) if use_peft else None

        
        if peft_config is not None:  
            peft_config = LoraConfig(**peft_config)

        return cls(
            llm_name_or_path=llm_name_or_path,
            # rag configs
            rag_mode=rag_mode,
            pos_shots_num=pos_shots_num,
            neg_shots_num=neg_shots_num,
            insights_num=insights_num,
            embedding_model_name_or_path=embedding_model_name_or_path,
            rag_working_dir=os.path.join(working_dir, 'rag'),
            database_dir=database_dir,

            # weaver configs
            use_weaver=use_weaver,
            latents_len=latents_len,
            peft_config=peft_config,

        )