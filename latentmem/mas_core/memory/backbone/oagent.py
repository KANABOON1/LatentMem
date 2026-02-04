import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterable

from langchain_chroma import Chroma
from langchain.docstore.document import Document
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from latentmem.utils.message import Trajectory
from latentmem.mas_core.memory.backbone import base_rag
from latentmem.mas_core.memory.backbone import prompt
from latentmem.mas_core.memory.backbone import utils

class OAgentMemory(base_rag.BaseRAGMemory):

    def __init__(
        self,
        pos_shots_num: int,
        neg_shots_num: int,
        insights_num: int,
        embedding_model_name_or_path: str,
        working_dir: str,
        llm_name_or_path: str,
        
        insights_activation_threshold: int = 20, 
        insights_update_interval: int = 20, 
        insights_update_iterations: int = 5,  

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
            do_sample=False,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        # insights related
        self.insights_num = insights_num
        self.insights_activation_threshold = insights_activation_threshold
        self.insights_update_interval = insights_update_interval
        self.insights_update_iterations = insights_update_iterations

        # insight layer
        self.insight_layer = InsightLayer(
            working_dir=self.working_dir,
            llm_model=self.llm,
            tokenizer=self.tokenizer,
            generation_common_settings=self.generation_common_settings,
            task_storage=self.main_memory,
        )

    @property
    def memory_size(self):
        num_records = self.main_memory.get()["ids"]
        return len(num_records)
    
    def add(self, mas_trajectory: Trajectory):
        
        meta_data: dict = Trajectory.to_serializable(mas_trajectory)
        memory_doc = Document(
            page_content=mas_trajectory.task_init_description,   
            metadata=meta_data
        )
        if mas_trajectory.label == True or mas_trajectory.label == False:
            self.main_memory.add_documents([memory_doc])
        else:
            raise ValueError('The mas_message must have label!')
        
        # finetune and merge insights
        if self.memory_size >= self.insights_activation_threshold and self.memory_size % self.insights_update_interval == 0:
            self.insight_layer.finetune_insights(self.insights_update_iterations)
    
    def _retrieve(
        self,
        query: str,
        pos_shots_num: int,
        neg_shots_num: int 
    ) -> tuple[list[Trajectory], list[Trajectory]]:
        pos_shots_doc: list[tuple[Document, float]] = []
        neg_shots_doc: list[tuple[Document, float]] = []
        
        if self.pos_shots_num != 0:
            pos_shots_doc = self.main_memory.similarity_search_with_score(
                query=query, k=pos_shots_num, filter={'label': True}
            )
        if self.neg_shots_num != 0:
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
        
        pos_shots, neg_shots = self._retrieve(query, self.pos_shots_num, self.neg_shots_num)

        insights_with_score = self.insight_layer.query_insights_with_score(query, self.insights_num)
        insights = [insight for insight, _ in insights_with_score][:self.insights_num]

        return base_rag.RAGMemoryResponse(
            pos_shots=pos_shots,
            neg_shots=neg_shots,
            extra_fields=dict(insights=insights)
        )

 
    
@dataclass
class InsightLayer:

    working_dir: str
    task_storage: Chroma

    llm_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    generation_common_settings: dict

    def __post_init__(self):

        self.persist_file: str = os.path.join(self.working_dir,f'insights.json')
        if not os.path.exists(self.persist_file):
            self.insights_memory = []
        else:
            with open(self.persist_file, encoding="utf-8") as f:
                self.insights_memory = json.load(f)

    def query_insights_with_score(self, task_query: str, top_k: int = None) -> list[tuple[str, float]]:
        SUCC_NUM, FAIL_NUM = 4, 2
        related_successful_tasks, related_failed_tasks = self._retrieve_memory(task_query, successful_topk=SUCC_NUM, failed_topk=FAIL_NUM)
        task_mains: list[str] = [task.task_init_description for task in related_successful_tasks + related_failed_tasks]
        task_mains.append(task_query)
        
        insights_score = defaultdict(float)
        for task_main in task_mains:
            _, related_insights = self._find_related_insights(task_mains=[task_main])
            for insight in related_insights:
                insights_score[insight.get('rule')] += 1  
        
        sorted_insights = sorted(insights_score.items(), key=lambda x: x[1], reverse=True) 
        if top_k is not None:
            sorted_insights = sorted_insights[:top_k]

        return sorted_insights

    def clear_insights(self):
        self.insights_memory = [self.insights_memory[i] for i in range(len(self.insights_memory)) 
                        if self.insights_memory[i]['score'] > 0] 

    def _retrieve_memory(
        self,
        query_task: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1
    ) -> tuple[list[Trajectory], list[Trajectory]]:

        true_tasks_doc: list[tuple[Document, float]] = []
        false_tasks_doc: list[tuple[Document, float]] = []

        if successful_topk != 0:
            true_tasks_doc = self.task_storage.similarity_search_with_score(
                query=query_task, k=successful_topk, filter={'label': True}
            )
        if failed_topk != 0:
            false_tasks_doc = self.task_storage.similarity_search_with_score(
                query=query_task, k=failed_topk, filter={'label': False}
            )
        sorted(true_tasks_doc, key=lambda x: x[1]) 
        sorted(false_tasks_doc, key=lambda x: x[1]) 

        true_task_messages: list[Trajectory] = []
        false_task_messages: list[Trajectory] = []
        for doc in true_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: Trajectory = Trajectory.from_serializable(meta_data)
            true_task_messages.append(mas_message)
        
        for doc in false_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: Trajectory = Trajectory.from_serializable(meta_data)
            false_task_messages.append(mas_message)

        return true_task_messages, false_task_messages
    
    @property
    def task_size(self):
        num_records = self.task_storage.get()["ids"]
        return len(num_records)
    
    def _find_related_insights(
        self,
        task_mains: list[str],
        threshold: float = 1
    ) -> tuple[list[int], list[dict]]:

        rule_set: list[tuple[dict, int, int]] = []  # (rule, score, index)

        for idx, rule in enumerate(self.insights_memory):
            score: int = sum(task in rule.get('positive_correlation_tasks', []) for task in task_mains)
            if score >= threshold:
                rule_set.append((rule, score, idx))

        rule_set.sort(key=lambda x: x[1], reverse=True)

        rule_indices = [item[2] for item in rule_set]
        sorted_rules = [item[0] for item in rule_set]

        return rule_indices, sorted_rules
    
    def finetune_insights(self, num_points: int):
        
        for _ in range(num_points):  
            
            random_id = random.choice(self.task_storage.get()['ids'])
            random_entry = self.task_storage.get(ids=[random_id])
            if 'metadatas' in random_entry and random_entry['metadatas']:
                random_metadata = random_entry['metadatas'][0]  
            else:
                raise RuntimeError('Incomplete data.')
            mas_message: Trajectory = Trajectory.from_serializable(random_metadata)
            
            SUCCESS_TASK_NUM, FAIL_TASK_NUM = 3, 1
            pos_shots, neg_shots = self._retrieve_memory(
                query_task=mas_message.task_init_description, successful_topk=SUCCESS_TASK_NUM, failed_topk=FAIL_TASK_NUM
            )
            if mas_message.label == True:
                pos_shots.append(mas_message)
            else:
                neg_shots.append(mas_message)
            all_task_mains: list[str] = [traj.task_init_description for traj in pos_shots + neg_shots]
            
            related_insight_ids, _ = self._find_related_insights(all_task_mains, len(all_task_mains) / 2)

            self._finetune_insights(pos_shots, neg_shots, related_insight_ids)
        
        self.clear_insights()
        self._index_done()


    def _finetune_insights(
        self,
        pos_shots: list[Trajectory],
        neg_shots: list[Trajectory],
        insight_ids: list[int]
    ) -> None:

        def map_operations(origin_operations: list[tuple]) -> list[tuple]:
            processed_operations: list[tuple] = []
            for (operation, text) in origin_operations:
                res: list = operation.split(' ')
                
                if "ADD" in res:
                    operation = "ADD"

                elif len(res) == 2:
                    
                    if len(insight_ids) == 0:    
                        continue

                    insight_id: int = int(res[1]) - 1  # 
                    if insight_id >= len(insight_ids) or insight_id < 0:
                        continue
                    
                    res[1] = str(insight_ids[insight_id] + 1)   
                    operation: str = ' '.join(res)

                processed_operations.append((operation, text))
            
            return processed_operations
        
        rule_list: list[dict] = [self.insights_memory[i] for i in insight_ids]
        
        compare_pairs: list[tuple[Trajectory, Trajectory]] = []
        for id, fail_task in enumerate(neg_shots):
            if id >= len(pos_shots):
                break
            success_task = pos_shots[id]
            compare_pairs.append((success_task, fail_task))

        successful_task_chunks: list[list[Trajectory]] = utils.random_divide_list(pos_shots, 5) 
        
        MAX_RULE_THRESHOLD: int = 10
        suffix: str = prompt.GMEMORY_FINETUNE_INSIGHTS_SUFFIX['full'] if len(self.insights_memory) > MAX_RULE_THRESHOLD \
                      else prompt.GMEMORY_FINETUNE_INSIGHTS_SUFFIX['not_full']
        
        generation_config = GenerationConfig(**self.generation_common_settings, max_new_tokens=100)

        for pair in compare_pairs:
            compare_prompt = self._build_comparative_prompts(pair[0], pair[1], rule_list)
            compare_prompt[1]["content"] = compare_prompt[1]["content"] + suffix  
            response = utils.llm_generate(self.llm_model, self.tokenizer, generation_config, [compare_prompt])[0]
            
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            
            self._update_rules(
                [pair[0].task_init_description, pair[1].task_init_description], 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
        
        for chunk in successful_task_chunks:
            success_prompt = self._build_success_prompts(chunk, rule_list) 
            success_prompt[1]["content"] = success_prompt[1]["content"] + suffix   
            response = utils.llm_generate(self.llm_model, self.tokenizer, generation_config, [success_prompt])[0]
            
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            
            self._update_rules(
                [traj.task_init_description for traj in chunk], 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
        
        self.clear_insights()
        self._index_done()

    def _index_done(self):
        with open(self.persist_file, "w", encoding="utf-8") as f:
            json.dump(self.insights_memory, f, indent=2, ensure_ascii=False, separators=(",", ": "))

    def _build_comparative_prompts(self, pos_shot: Trajectory, neg_shot: Trajectory, insights: list[dict]) -> list[dict]:
        
        existing_rules: list[str] = [insight['rule'] for insight in insights]

        if len(existing_rules) == 0:
            rule_text = "No insights."
        else:
            rule_text = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])

        user_prompt = prompt.GMEMORY_COMPARE_GENERATION_USER_PROMPT.format(   
            pos_shot=pos_shot.to_text(),   
            neg_shot=neg_shot.to_text(),
            existing_rules=rule_text
        )

        return [
            {"role": "system", "content": prompt.GMEMORY_COMPARE_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ] 
    
    def _build_success_prompts(
        self,
        pos_shots: Iterable[Trajectory],
        insights: list[dict],
    ) -> list[dict]:
         
        existing_rules: list[str] = [insight['rule'] for insight in insights]

        if len(existing_rules) == 0:
            rule_text = "No insights."
        else:
            rule_text = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])
        
        history = [traj.to_text() for traj in pos_shots]
        user_prompt = prompt.GMEMORY_SUMMARIZE_GENERATION_USER_PROMPT.format(
            pos_shots='\n\n'.join(history),
            existing_rules=rule_text
        )

        return [
            {"role": "system", "content": prompt.GMEMORY_SUMMARIZE_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content":  user_prompt}
        ] 
    
    def _parse_rules(self, llm_text):
        pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
        matches = re.findall(pattern, llm_text)

        res = []
        banned_words = ['ADD', 'AGREE', 'EDIT']
        for operation, text in matches:
            text = text.strip()
            if text != '' and not any([w in text for w in banned_words]) and text.endswith('.'):

                if 'ADD' in operation:
                    res.append(('ADD', text))
                else:
                    res.append((operation.strip(), text))
        return res
    
    def _update_rules(
        self,
        relative_tasks: list[str],
        operations: list[tuple[str, str]], 
        max_rules_num: int = 10
    ) -> None:

        delete_indices = []
        for i in range(len(operations)):
            operation, operation_rule_text = operations[i]
            operation_type = operation.split(' ')[0]
            rule_num = int(operation.split(' ')[1]) if ' ' in operation else None

            if operation_type == 'ADD':    
                if self._is_existing_rule(operation_rule_text): 
                    delete_indices.append(i)
                    
            elif operation_type == 'EDIT':   
                if self._is_existing_rule(operation_rule_text): 
                    rule_num: int = self._retrieve_rule_index(operation_rule_text)
                    operations[i] = (f'AGREE {rule_num + 1}', operation_rule_text)   

                elif (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
                        
            elif operation_type == 'REMOVE' or operation_type == 'AGREE':  
                if (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
            
            else: 
                delete_indices.append(i)

        operations = [operations[i] for i in range(len(operations)) if i not in delete_indices] 
        

        list_full: bool = len(self.insights_memory) >= max_rules_num  
        for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']: 
            for i in range(len(operations)):
                operation, operation_rule_text = operations[i]
                operation_type = operation.split(' ')[0]
                if operation_type != op:
                    continue

                if operation_type == 'REMOVE': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    remove_strength = 3 if list_full else 1
                    rule_data['score'] -= remove_strength
                    rule_data['negative_correlation_tasks'] = list(set(rule_data['negative_correlation_tasks'] + relative_tasks))  

                elif operation_type == 'AGREE':
                    rule_index: int = self._retrieve_rule_index(operation_rule_text) 
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'EDIT': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['rule'] = operation_rule_text
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'ADD': 
                    meta_data: dict = {
                        'rule': operation_rule_text,
                        'score': 2,         
                        'positive_correlation_tasks': list(relative_tasks),
                        'negative_correlation_tasks': list()
                    }
                    self.insights_memory.append(meta_data)

    def _is_existing_rule(self, operation_rule_text: str) -> bool:

        for insight in self.insights_memory:
            if insight['rule'] in operation_rule_text:
                return True
        return False
    
    def _retrieve_rule_index(self, operation_rule_text: str) -> int:

        for idx, insight in enumerate(self.insights_memory):
            if insight['rule'] in operation_rule_text:
                return idx
        return -1