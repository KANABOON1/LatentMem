import logging
from typing import Optional

import torch
from transformers import GenerationConfig, AutoModelForCausalLM
import uuid

from common.registry import registry
from common.utils.tensor_utils import fix_model_parameters
from latentmem.mas_core.base_centralized_memory import BaseCentralizedMemory
from latentmem.mas_core.base_memory_mas import BaseMemoryMAS
from latentmem.mas_core.structures.autogen.prompts import prompt
from latentmem.utils.agent import LLMAgent
from latentmem.utils.message import (
    MessageNode,
    MessageGraph,
)

@registry.register_mas("autogen")
class AutoGenMemoryMAS(BaseMemoryMAS):
    def __init__(
        self, 
        llm_name_or_path: str, 
        centralized_memory: Optional[BaseCentralizedMemory] = None,
        share_llm: bool = True,
        task_domain: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            centralized_memory=centralized_memory,
            llm_name_or_path=llm_name_or_path,
            share_llm=share_llm,
            task_domain=task_domain
        )
        
        assistant_model = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        
        if share_llm:  
            user_proxy_model = assistant_model
        else:    
            user_proxy_model = AutoModelForCausalLM.from_pretrained(
                llm_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
        
        fix_model_parameters(assistant_model)
        fix_model_parameters(user_proxy_model)
        
        domain_prompt = prompt.get_domain_prompt(self.task_domain)
        
        # A person is the sum of their contexts.
        self.assistant_agent = LLMAgent(
            model=assistant_model, 
            tokenizer=self.shared_tokenizer, 
            role="assistant agent",
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=domain_prompt.ASSISTANT_SYSTEM_PROMPT_TEMPLATE, 
            user_prompt_template=domain_prompt.ASSISTANT_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )
        self.user_proxy_agent = LLMAgent(
            model=user_proxy_model, 
            tokenizer=self.shared_tokenizer,
            role="user proxy agent", 
            id=str(uuid.uuid4()),
            topology_node_id=1,
            system_prompt_template=domain_prompt.USER_PROXY_SYSTEM_PROMPT_TEMPLATE, 
            user_prompt_template=domain_prompt.USER_PROXY_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )

        self.agents_list.extend([self.assistant_agent, self.user_proxy_agent])
        self.centralized_memory.register_agents(self.agents_list)

        self.config = assistant_model.config
            
    def generate(self, task_domain_instructions: list[str], user_inputs: list[str], generation_config: GenerationConfig) -> list[MessageGraph]:

        batch_message_graphs = [MessageGraph(state=input) for input in user_inputs]

        # --- actor outputs ---
        assistant_user_inputs, assistant_system_inputs = dict(),  dict()
        assistant_user_inputs["task_description"] = user_inputs
        assistant_system_inputs["task_domain_instructions"] = task_domain_instructions
        assistant_messages = self.assistant_agent.invoke(assistant_system_inputs, assistant_user_inputs, generation_config)
        
        for ast_msg, mas_graph in zip(assistant_messages, batch_message_graphs):
            mas_graph.update_message_graph(ast_msg, self.assistant_agent.role, None) 
            mas_graph.action = ast_msg.response

        # --- critic outputs ---
        userproxy_user_inputs, userproxy_system_inputs = dict(), dict()
        userproxy_user_inputs["task_description"] = assistant_user_inputs["task_description"]
        userproxy_user_inputs["assistant_output"] = [msg.response for msg in assistant_messages] 
        userproxy_system_inputs["task_domain_instructions"] = task_domain_instructions
        user_proxy_messages = self.user_proxy_agent.invoke(userproxy_system_inputs, userproxy_user_inputs, generation_config)
        
        for i, msg in enumerate(user_proxy_messages):
            batch_message_graphs[i].update_message_graph(msg, self.user_proxy_agent.role, [self.assistant_agent.role])
            batch_message_graphs[i].action = msg.response

        return batch_message_graphs

        
            