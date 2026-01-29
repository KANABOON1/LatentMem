from common.utils.code_utils import PyExecutor, extract_python_code, rename_function
from data.base_env import StaticEnv

KODCODE_INSTRUCTIONS = "Write a Python function according to the given task"

class KodCodeEnv(StaticEnv):

    def __init__(self, config):
        super().__init__(config)


    def set_env(self, task_config: dict) -> tuple[str, str]:
        
        self.prompt = task_config.get("prompt")  
        self.test = task_config.get("test")  
        self.test_info = task_config.get("test_info") 

        return KODCODE_INSTRUCTIONS, self.prompt
    
    def step(self, action: str) -> tuple[str, str]:

        self.reward = self.compute_reward(completions=[action], test=[self.test], test_info=[self.test_info])[0]  
        
        if self.reward == 1.0:
            self.observation = "True!"
            self.done = True
        else:
            self.observation = "False!"
            self.done = False

        return self.observation, self.reward, self.done

    @classmethod
    def compute_reward(cls, completions: list[str], test: list[str], test_info: list, **kwargs) -> list[float]:
 
        py_executor = PyExecutor()
        scores = []
        for completion, t, tf in zip(completions, test, test_info): 
            func_blocks = extract_python_code(completion.strip())
            collected_answer = '\n'.join(func_blocks)
            renamed_answer = rename_function(collected_answer, tf[0]["function_name"])
            _, _, results = py_executor.execute(renamed_answer, [t])

            score = sum(results) / len(results)  
            scores.append(score)
        
        return scores
