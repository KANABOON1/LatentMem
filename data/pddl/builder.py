from datasets import DatasetDict, Dataset
import json
 
from common.registry import registry
from data.base_builder import BaseDataBuilder
from data.pddl.env.pddl_env import PDDLEnv

PDDL_SYSTEM_MESSAGE = ""

def get_all_environment_configs(game_names: list[str], label_path: str):
    def load_annotation(path):
        all_annotations = None  
        difficulty = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                line = json.loads(line.strip())
                if "difficulty" in line:
                    difficulty.append(line["difficulty"])
                else:
                    raise ValueError("No difficulty in annotation file")
        return all_annotations, difficulty
    
    Num_Problems = {
        "barman":20, "blockworld":10, "gripper":20, "tyreworld":10
    }
    env_configs = []
    iter_num = 0
    _, difficulties = load_annotation(label_path)
    
    for game_name in game_names:
        num_problems = Num_Problems[game_name]
        for i in range(num_problems):
            env_configs.append({
                "game_name": game_name,
                "problem_index": i,
                "difficulty": difficulties[iter_num]
            })
            
            iter_num += 1
    
    return env_configs

@registry.register_builder("pddl")
class PDDLBuilder(BaseDataBuilder): 
    
    def get_env_cls(self):
        return PDDLEnv
 
    def _build_datasets(self) -> DatasetDict:        
        
        TASK_NAMES = ["barman", "blockworld", "gripper", "tyreworld"]
        pddl_tasks: list[dict] = get_all_environment_configs(TASK_NAMES, 'data/pddl/test.jsonl')
        test_dataset = Dataset.from_list(pddl_tasks)

        empty_dataset = Dataset.from_dict({k: [] for k in test_dataset.column_names})

        dataset_dict = DatasetDict({
            "train": empty_dataset,
            "valid": empty_dataset,
            "test": test_dataset
        })

        return dataset_dict