from langchain.docstore.document import Document

from memmaster.utils.message import Trajectory

from memmaster.mas_core.memory.backbone import base_rag

class MetaGPT(base_rag.BaseRAGMemory):

    def __init__(
        self,
        pos_shots_num: int,
        neg_shots_num: int,
        embedding_model_name_or_path: str,
        llm_name_or_path: str,
        working_dir: str,
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
    