import copy
from datasets import DatasetDict, load_dataset
from typing import Dict, List

from common.registry import registry
from data.base_builder import BaseDataBuilder
from data.triviaqa.env import TriviaQAEnv


TRIVIAQA_SYSTEM_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. \
"""

@registry.register_builder("triviaqa")
class TriviaQABuilder(BaseDataBuilder):  # Env

    def get_env_cls(self):
        return TriviaQAEnv

    def _build_datasets(self) -> DatasetDict:
        
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")   
        raw_train_dataset = ds["train"].select(range(5000))  
        raw_valid_dataset = ds["validation"].select(range(1000))   
        raw_test_dataset = ds["validation"].select(range(1000, len(ds["validation"])))
       
        num_workers = 32
        train_dataset = raw_train_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        valid_dataset = raw_valid_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        test_dataset = raw_test_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())

        dataset_dict = DatasetDict()
        dataset_dict["train"] = train_dataset
        dataset_dict["valid"] = valid_dataset
        dataset_dict["test"] = test_dataset

        return dataset_dict

    @classmethod
    def _preprocess(cls, example: Dict) -> Dict:
        output = copy.deepcopy(example)
        output["answer"] = output["answer"]["normalized_aliases"]
        output["prompt"] = output["question"]
        return output

    @classmethod
    def _keep_keys(cls) -> List[str]:
        return ["prompt", "answer"]