import re

import torch
from transformers import GenerationConfig
from langchain.docstore.document import Document

from latentmem.utils.message import Trajectory

from latentmem.mas_core.memory.backbone import base_rag
from latentmem.mas_core.memory.backbone import prompt

class Voyager(base_rag.BaseRAGMemory):

    def __init__(
        self,
        pos_shots_num: int,
        neg_shots_num: int,
        embedding_model_name_or_path: str,
        working_dir: str,
        llm_name_or_path: str,
        **kwargs
    ):  
        super().__init__(
            pos_shots_num=pos_shots_num,
            neg_shots_num=neg_shots_num,
            embedding_model_name_or_path=embedding_model_name_or_path,
            llm_name_or_path=llm_name_or_path,
            working_dir=working_dir, 
            **kwargs
        )

        self.generation_config = GenerationConfig(
            do_sample=True, 
            temperature=1.0, 
            max_new_tokens=50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )

    def add(self, mas_trajectory: Trajectory):
        assert mas_trajectory.label == True or mas_trajectory.label == False
        assert mas_trajectory is not None
        assert mas_trajectory.task_init_description is not None

        trajectory_meta_data = mas_trajectory.to_serializable()
        
        user_prompt_content = prompt.VOYAGER_USER_PROMPT.format(
            trajectory=mas_trajectory.to_text()
        )

        conversations = [
            {"role": "system", "content": prompt.VOYAGER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_content}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            conversations, 
            tokenize=True, 
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.llm.device) 
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        input_len = input_ids.shape[1] 

        generated_ids = self.llm.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
        )

        output_ids = generated_ids[0][input_len:] 

        generated_page_content = self.tokenizer.decode(
            output_ids, 
            skip_special_tokens=True
        ).strip() 

        document = Document(
            page_content=generated_page_content, 
            metadata=trajectory_meta_data
        )

        self.main_memory.add_documents([document])


    def retrieve(self, query: str, **kwargs) -> base_rag.RAGMemoryResponse:
        
        pos_shots_doc: list[tuple[Document, float]] = []
        neg_shots_doc: list[tuple[Document, float]] = []
        
        if self.pos_shots_num != 0:
            pos_shots_doc = self.main_memory.similarity_search_with_score(
                query=query, k=self.pos_shots_num, filter={'label': True}
            )
        if self.neg_shots_num != 0:
            neg_shots_doc = self.main_memory.similarity_search_with_score(
                query=query, k=self.neg_shots_num, filter={'label': False}
            )

        pos_shots_doc.sort(key=lambda x: x[1])
        neg_shots_doc.sort(key=lambda x: x[1])
        
        pos_shots: list[Trajectory] = []
        neg_shots: list[Trajectory] = []
        for doc in pos_shots_doc:
            meta_data: dict = doc[0].metadata
            mas_message = Trajectory.from_serializable(meta_data)
            pos_shots.append(mas_message)
        
        for doc in neg_shots_doc:
            meta_data: dict = doc[0].metadata
            mas_message: Trajectory = Trajectory.from_serializable(meta_data)
            neg_shots.append(mas_message)

        memory_response = base_rag.RAGMemoryResponse(
            pos_shots=pos_shots, neg_shots=neg_shots
        )

        return memory_response
