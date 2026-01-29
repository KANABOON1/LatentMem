import re

import torch
from transformers import GenerationConfig
from langchain.docstore.document import Document

from memmaster.utils.message import Trajectory

from memmaster.mas_core.memory.backbone import base_rag
from memmaster.mas_core.memory.backbone import prompt

class Generative(base_rag.BaseRAGMemory):

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
            max_new_tokens=20,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )


    def add(self, mas_trajectory: Trajectory):
        assert mas_trajectory.label == True or mas_trajectory.label == False
        assert mas_trajectory is not None
        assert mas_trajectory.task_init_description is not None

        trajectory_meta_data = mas_trajectory.to_serializable()

        document = Document(
            page_content=mas_trajectory.task_init_description,
            metadata=trajectory_meta_data
        )

        self.main_memory.add_documents([document])
    
    def _retrieve(self, query: str, pos_shots_num: int, neg_shots_num: int) -> tuple[list[Trajectory], list[Trajectory]]:
        pos_shots_doc: list[tuple[Document, float]] = []
        neg_shots_doc: list[tuple[Document, float]] = []
        
        if pos_shots_num != 0:
            pos_shots_doc = self.main_memory.similarity_search_with_score(
                query=query, k=pos_shots_num, filter={'label': True}
            )
        if neg_shots_num != 0:
            neg_shots_doc = self.main_memory.similarity_search_with_score(
                query=query, k=neg_shots_num, filter={'label': False}
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
        
        return pos_shots, neg_shots

    def retrieve(self, query: str, **kwargs) -> base_rag.RAGMemoryResponse:
        
        pos_shots_candidate, neg_shots_candidate = self._retrieve(
            query, 2 * self.pos_shots_num, 2 * self.neg_shots_num
        )

        all_candidate_shots = pos_shots_candidate + neg_shots_candidate
        
        if not all_candidate_shots:
            memory_response = base_rag.RAGMemoryResponse(
                pos_shots=[], neg_shots=[]
            )
            return memory_response
        
        prompts = []
        for shot in all_candidate_shots:
            prompt_text = self._create_scoring_prompt(query, shot)
            prompts.append(prompt_text)

        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left",
            truncation=True
        ).to(self.llm.device) 
        
        with torch.no_grad():
            generated_ids = self.llm.generate(
                **inputs,
                generation_config=self.generation_config,
            )
        
        input_len = inputs.input_ids.shape[1]
        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids[:, input_len:], 
            skip_special_tokens=True
        )
        
        scores = []
        for output_text in decoded_outputs:
            score = self._parse_score_from_output(output_text)
            scores.append(score)

        scored_candidates = list(zip(all_candidate_shots, scores))
        
        scored_pos_candidates: list[tuple[Trajectory, float]] = []
        scored_neg_candidates: list[tuple[Trajectory, float]] = []

        for shot, score in scored_candidates:
            if score != -1.0: 
                if shot.label is True:
                    scored_pos_candidates.append((shot, score))
                elif shot.label is False:
                    scored_neg_candidates.append((shot, score))
        
        scored_pos_candidates.sort(key=lambda x: x[1], reverse=True)
        scored_neg_candidates.sort(key=lambda x: x[1], reverse=True)
        
        final_pos_shots = [shot for shot, score in scored_pos_candidates][:self.pos_shots_num]
        final_neg_shots = [shot for shot, score in scored_neg_candidates][:self.neg_shots_num]

        memory_response = base_rag.RAGMemoryResponse(
            pos_shots=final_pos_shots, neg_shots=final_neg_shots
        )

        return memory_response
    
    def _create_scoring_prompt(self, query: str, shot: Trajectory) -> str:
        system_prompt = prompt.GENERATIVE_SYSTEM_PROMPT.format()
        user_prompt = prompt.GENERATIVE_USER_PROMPT.format(
            task_description=query,
            trajectory=shot.to_text()
        )
        conversations = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]
        
        full_prompt = self.tokenizer.apply_chat_template(
            conversations, 
            add_generation_prompt=True,
            tokenize=False
        )
        return full_prompt


    def _parse_score_from_output(self, llm_output: str) -> float:
        cleaned_output = llm_output.strip()
      
        match = re.fullmatch(r"(\d{1,2})", cleaned_output)

        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                pass
        
        return -1.0