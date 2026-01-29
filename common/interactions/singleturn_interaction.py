from transformers import GenerationConfig

from memmaster.mas_core.base_memory_mas import BaseMemoryMAS
from memmaster.utils.message import Trajectory
from common.interactions.base_interaction import (
    InteractionConfig, 
    InteractionManager,
    InteractionDataProto
)

class SingleTurnInteractionManager(InteractionManager):
    def __init__(
        self,
        memory_mas: BaseMemoryMAS,
        interaction_config: InteractionConfig,
        generation_config: GenerationConfig
    ):  
        super().__init__(memory_mas, interaction_config, generation_config)      

    def run_inter_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:

        # preprocess: clip the prompt length
        domain_instructions = gen_batch.no_tensor_batch["domain_instructions"]
        task_descriptions = gen_batch.no_tensor_batch["task_descriptions"]
        envs = gen_batch.no_tensor_batch["envs"]
        
        # call mas to generate
        message_graphs = self.memory_mas.generate(domain_instructions, task_descriptions, self.generation_config) 

        trajectories: list[Trajectory] = []
        
        for task_description, message_graph, env in zip(task_descriptions, message_graphs, envs):
            
            trajectory = Trajectory()
            trajectory.task_init_description = task_description
            trajectory.trajectory = [message_graph] 
            _, trajectory.reward, trajectory.label = env.step(message_graph.action)

            trajectories.append(trajectory)

        no_tensor_batch = dict(trajectories=trajectories)
        
        return InteractionDataProto(no_tensor_batch=no_tensor_batch)


