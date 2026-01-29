import re

from langchain.docstore.document import Document
from transformers import GenerationConfig

from memmaster.utils.message import Trajectory
from memmaster.mas_core.memory.backbone import base_rag
from memmaster.mas_core.memory.backbone import prompt
from memmaster.mas_core.memory.backbone import utils

class Master(base_rag.BaseRAGMemory):
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

        self.generation_common_settings = dict(
            do_sample=True,
            temperature=1.0,
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
    
    def _retrieve(
        self, 
        query: str, 
        pos_shots_num: int, 
        neg_shots_num: int, 
    ) -> tuple[list[Trajectory], list[Trajectory]]:
        
        pos_shots_doc: list[tuple[Document, float]] = []
        neg_shots_doc: list[tuple[Document, float]] = []

        def fetch_and_filter(label: bool, k: int) -> list[tuple[Document, float]]:
            if k == 0:
                return []

            initial_k = k * 2
            docs_with_score = self.main_memory.similarity_search_with_score(
                query=query, k=initial_k, filter={'label': label}
            )
            docs_with_score.sort(key=lambda x: x[1])
            
            final_docs: list[tuple[Document, float]] = []
            
            for doc, score in docs_with_score:
                if doc.page_content == query:
                    continue
                
                final_docs.append((doc, score))
                
                if len(final_docs) >= k:
                    break
            
            return final_docs
        
        pos_shots_doc = fetch_and_filter(label=True, k=pos_shots_num)
        neg_shots_doc = fetch_and_filter(label=False, k=neg_shots_num)

        pos_shots_doc.sort(key=lambda x: x[1])
        neg_shots_doc.sort(key=lambda x: x[1])

        pos_shots: list[Trajectory] = []
        neg_shots: list[Trajectory] = []
        
        for doc, _ in pos_shots_doc:
            meta_data: dict = doc.metadata
            mas_message = Trajectory.from_serializable(meta_data)
            pos_shots.append(mas_message)
        
        for doc, _ in neg_shots_doc:
            meta_data: dict = doc.metadata
            mas_message: Trajectory = Trajectory.from_serializable(meta_data)
            neg_shots.append(mas_message)
        
        return pos_shots, neg_shots
    
    def _generative_retrieve(
        self, query: str, pos_shots_num: int, neg_shots_num: int
    ) -> tuple[list[Trajectory], list[Trajectory]]:
        pos_shots_candidate, neg_shots_candidate = self._retrieve(query, 2 * pos_shots_num, 2 * neg_shots_num)
        all_candidate_shots = pos_shots_candidate + neg_shots_candidate
        
        if not all_candidate_shots:
            return [], []
        
        batch_conversations = []
        for shot in all_candidate_shots:
            conversation = self._create_scoring_prompt(query, shot)
            batch_conversations.append(conversation)

        generation_config = GenerationConfig(**self.generation_common_settings, max_new_tokens=20)
        outputs = utils.llm_generate(self.llm, self.tokenizer, generation_config, batch_conversations)
        
        scores = []
        for output_text in outputs:
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

        return final_pos_shots, final_neg_shots

    def _extract_agent_context(self, shots: list[Trajectory], agent_role: str):
        for shot in shots:
            trajectory = shot.trajectory
            contexts: list[str] = []

            for message_graph in trajectory:
                agent_info = message_graph.retrieve_message_graph(agent_role)
                agent_message = agent_info["message_node"]
                if agent_message is not None:
                    response = agent_message.response
                    context = response
                    contexts.append(context)
            
            contexts_str = '\n'.join(contexts)
            shot.add_extra_fields(agent_role, contexts_str)

    def retrieve(self, query: str, agent_role: str, **kwargs) -> base_rag.RAGMemoryResponse:

        pos_shots, neg_shots = self._generative_retrieve(
            query, self.pos_shots_num, self.neg_shots_num
        )

        all_shots = pos_shots + neg_shots
        self._extract_agent_context(all_shots, agent_role)

        return base_rag.RAGMemoryResponse(
            pos_shots=pos_shots, neg_shots=neg_shots
        )

    def _create_scoring_prompt(self, query: str, shot: Trajectory) -> list[dict]:
        system_prompt = prompt.GENERATIVE_SYSTEM_PROMPT.format()
        user_prompt = prompt.GENERATIVE_USER_PROMPT.format(
            task_description=query,
            trajectory=shot.to_text()
        )
        conversations = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]
        
        return conversations


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