import logging
from typing import Optional, Literal

import torch
from transformers import GenerationConfig, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
import uuid

from common.registry import registry
from common.utils.tensor_utils import fix_model_parameters
from memmaster.mas_core.base_centralized_memory import BaseCentralizedMemory
from memmaster.mas_core.base_memory_mas import BaseMemoryMAS
from memmaster.mas_core.structures.macnet.prompts import prompt
from memmaster.utils.agent import LLMAgent
from memmaster.utils.message import (
    MessageGraph,
)



@registry.register_mas("macnet")
class MacNetMemoryMAS(BaseMemoryMAS):
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
        
        assert share_llm
        shared_model = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        fix_model_parameters(shared_model)

        self.domain_prompt = prompt.get_domain_prompt(self.task_domain)
        
        self.actor_agent_1 = LLMAgent(
            model=shared_model, 
            tokenizer=self.shared_tokenizer, 
            role="actor agent 1", 
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=self.domain_prompt.ACTOR_SYSTEM_PROMPT_TEMPLATE, 
            user_prompt_template=self.domain_prompt.ACTOR_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )
        self.actor_agent_2 = LLMAgent(
            model=shared_model, 
            tokenizer=self.shared_tokenizer, 
            role="actor agent 2",  
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=self.domain_prompt.ACTOR_SYSTEM_PROMPT_TEMPLATE, 
            user_prompt_template=self.domain_prompt.ACTOR_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )

        self.critic_agent_1 = LLMAgent(
            model=shared_model,
            tokenizer=self.shared_tokenizer, 
            role="critic agent 1", 
            id=str(uuid.uuid4()),
            topology_node_id=1,
            system_prompt_template=self.domain_prompt.CRITIC_SYSTEM_PROMPT_TEMPLATE, 
            user_prompt_template=self.domain_prompt.CRITIC_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory            
        )

        self.critic_agent_2 = LLMAgent(
            model=shared_model,
            tokenizer=self.shared_tokenizer, 
            role="critic agent 2",
            id=str(uuid.uuid4()),
            topology_node_id=1,
            system_prompt_template=self.domain_prompt.CRITIC_SYSTEM_PROMPT_TEMPLATE, 
            user_prompt_template=self.domain_prompt.CRITIC_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory            
        )

        self.summarizer_agent = LLMAgent(
            model=shared_model,
            tokenizer=self.shared_tokenizer, 
            role="summarizer agent",
            id=str(uuid.uuid4()),
            topology_node_id=2,
            system_prompt_template=self.domain_prompt.SUMMARIZER_SYSTEM_PROMPT_TEMPLATE, 
            user_prompt_template=self.domain_prompt.SUMMARIZER_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory                       
        )

        self.agents_list.extend([
            self.actor_agent_1, 
            self.actor_agent_2,
            self.critic_agent_1,
            self.critic_agent_2,
            self.summarizer_agent
        ])
        self.centralized_memory.register_agents(self.agents_list)

        self.config = shared_model.config

    def generate(self, task_domain_instructions: list[str], user_inputs: list[str], generation_config: GenerationConfig) -> list[MessageGraph]:
        
        batch_message_graphs = [MessageGraph(state=input) for input in user_inputs]

        # --- actor 1 outputs ---
        actor1_user_inputs, actor1_system_inputs = dict(),  dict()
        actor1_user_inputs["task_description"] = user_inputs
        actor1_system_inputs["task_domain_instructions"] = task_domain_instructions
        actor1_messages = self.actor_agent_1.invoke(actor1_system_inputs, actor1_user_inputs, generation_config)
        
        for act_msg, mas_graph in zip(actor1_messages, batch_message_graphs):
            mas_graph.update_message_graph(act_msg, self.actor_agent_1.role, None) 

        # --- critic 1 outputs ---
        critic1_user_inputs, critic1_system_inputs = dict(), dict()
        critic1_user_inputs["task_description"] = user_inputs
        critic1_user_inputs["actor_output"] = [msg.response for msg in actor1_messages]
        critic1_system_inputs["task_domain_instructions"] = task_domain_instructions
        critic1_messages = self.critic_agent_1.invoke(critic1_system_inputs, critic1_user_inputs, generation_config)

        for cri_msg, mas_graph in zip(critic1_messages, batch_message_graphs):
            mas_graph.update_message_graph(cri_msg, self.critic_agent_1.role, [self.actor_agent_1.role])      
        
        # --- actor 2 outputs ---
        actor2_user_inputs, actor2_system_inputs = dict(),  dict()
        actor2_user_inputs["task_description"] = user_inputs
        actor2_system_inputs["task_domain_instructions"] = task_domain_instructions
        actor2_messages = self.actor_agent_2.invoke(actor2_system_inputs, actor2_user_inputs, generation_config)
        
        for act_msg, mas_graph in zip(actor2_messages, batch_message_graphs):
            mas_graph.update_message_graph(act_msg, self.actor_agent_2.role, None) 

        # --- critic 2 outputs ---
        critic2_user_inputs, critic2_system_inputs = dict(), dict()
        critic2_user_inputs["task_description"] = user_inputs
        critic2_user_inputs["actor_output"] = [msg.response for msg in actor2_messages]
        critic2_system_inputs["task_domain_instructions"] = task_domain_instructions
        critic2_messages = self.critic_agent_2.invoke(critic2_system_inputs, critic2_user_inputs, generation_config)

        for cri_msg, mas_graph in zip(critic2_messages, batch_message_graphs):
            mas_graph.update_message_graph(cri_msg, self.critic_agent_2.role, [self.actor_agent_2.role])

        # --- summarizer outputs ---
        summarizer_user_inputs, summarizer_system_inputs = dict(), dict()
        summarizer_user_inputs["task_description"] = user_inputs
        summarizer_user_inputs["feedback_page1"] = [
            self.domain_prompt.FEEDBACK_PAGE.format(actor_output=act_msg.response, critic_output=cri_msg.response)
            for act_msg, cri_msg in zip(actor1_messages, critic1_messages)
        ]
        summarizer_user_inputs["feedback_page2"] = [
            self.domain_prompt.FEEDBACK_PAGE.format(actor_output=act_msg.response, critic_output=cri_msg.response)
            for act_msg, cri_msg in zip(actor2_messages, critic2_messages)
        ]
        summarizer_system_inputs["task_domain_instructions"] = task_domain_instructions
        summarizer_messages = self.summarizer_agent.invoke(summarizer_system_inputs, summarizer_user_inputs, generation_config)

        for sum_msg, mas_graph in zip(summarizer_messages, batch_message_graphs):
            mas_graph.update_message_graph(sum_msg, self.summarizer_agent.role, [self.critic_agent_1.role, self.critic_agent_2.role])
            mas_graph.action = sum_msg.response 
        
        return batch_message_graphs