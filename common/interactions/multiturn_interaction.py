import copy
import torch
from transformers import GenerationConfig

from common.interactions.base_interaction import (
    InteractionDataProto,
    InteractionConfig, 
    InteractionManager
)
from memmaster.mas_core.base_memory_mas import BaseMemoryMAS
from memmaster.utils.message import MessageGraph, Trajectory
    
class MultiTurnInteractionManager(InteractionManager):
    def __init__(
        self,
        memory_mas: BaseMemoryMAS,
        interaction_config: InteractionConfig,
        generation_config: GenerationConfig
    ):
        super().__init__(memory_mas, interaction_config, generation_config)      

    def run_inter_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        """Run main LLM generation loop (conversation format)."""
        batch_size = len(gen_batch.no_tensor_batch["domain_instructions"])
        
        rollings = gen_batch   
        rollings.no_tensor_batch["inter_histories"] = [[] for _ in range(batch_size)]
        rollings.no_tensor_batch["trajectories"] = [
            Trajectory(task_init_description=rollings.no_tensor_batch["task_descriptions"][i]) 
            for i in range(batch_size)
        ]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        
        for _ in range(self.interaction_config.max_turns):
            if not active_mask.sum():   
                break            
            mask_list = active_mask.tolist()  
            rollings_active = {
                k: [item for item, keep in zip(v, mask_list) if keep]
                for k, v in rollings.no_tensor_batch.items()
            }

            task_contexts = self._build_task_contexts(rollings_active)

            message_graphs = self.memory_mas.generate(
                rollings_active["domain_instructions"], 
                task_contexts, 
                self.generation_config
            )

            responses = [msg_graph.action for msg_graph in message_graphs]
            responses = self._postprocess_responses(responses, rollings_active["envs"])
            all_responses, all_message_graphs = self._example_level_pad(responses, message_graphs, active_mask)
            
            next_obs, dones = self._execute_predictions(rollings, all_responses, active_mask)
            processed_obs = self._postprocess_observations(next_obs) 
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            
            interaction_histories, trajectories = self._update_interaction_history(
                rollings, 
                all_responses, 
                all_message_graphs, 
                processed_obs
            )
            rollings.no_tensor_batch["inter_histories"] = interaction_histories
            rollings.no_tensor_batch["trajectories"] = trajectories
  
        for trajectory, env in zip(rollings.no_tensor_batch["trajectories"], rollings.no_tensor_batch["envs"]):
            reward = env.feedback()
            trajectory.reward = reward  
            trajectory.label = (reward == 1.0) 
        
        return rollings
    
    def _build_task_contexts(self, rollings: dict) -> list[str]:
        task_descriptions = rollings.get("task_descriptions")
        if task_descriptions is None:
            raise ValueError("")
        
        inter_histories = rollings.get("inter_histories")
        if inter_histories is None:
            raise ValueError("")
        
        conversations: list[list[dict]] = []
        for task_description, inter_history in zip(task_descriptions, inter_histories):
            init_prompt = [{"role": "user", "content": task_description}]
            conversations.append(init_prompt + inter_history)
        
        task_contexts = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=False
        )
        return task_contexts
    
    def _update_interaction_history(
        self, 
        rollings: InteractionDataProto, 
        responses: list[str], 
        message_graphs: list[MessageGraph], 
        observations: list[str]
    ) -> tuple[list[list[dict]], list[Trajectory]]:
        # update conversations
        inter_histories = copy.deepcopy(rollings.no_tensor_batch.get("inter_histories"))
        assert len(inter_histories) == len(responses) == len(observations)
        for inter_history, response, observation in zip(inter_histories, responses, observations):
            assistant_info = {"role": "assistant", "content": response}
            user_info = {"role": "user", "content": observation}
            
            inter_history.append(assistant_info)
            inter_history.append(user_info)
        
        # update trajectories
        trajectories = copy.deepcopy(rollings.no_tensor_batch.get("trajectories"))
        assert len(trajectories) == len(responses) == len(observations)
        for trajectory, message_graph, observation in zip(trajectories, message_graphs, observations):
            if message_graph is not None:  
                message_graph.observation = observation  
                trajectory.add_step(message_graph)
        
        return inter_histories, trajectories
    
    def _postprocess_responses(self, responses: list[str], envs: list) -> list[str]:
        processed_responses_str = []
        for r, env in zip(responses, envs):
            processed_r = env.preprocess_action(r)
            processed_responses_str.append(processed_r)

        return processed_responses_str


    def _example_level_pad(
        self, responses: list[str], message_graphs: list[MessageGraph], active_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[str]]: 
        assert active_mask.sum() == len(responses)
        assert len(responses) == len(message_graphs)
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        
        # Create masked response strings
        padded_responses = [""] * batch_size
        padded_message_graphs = [None] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses[i] = responses[s]
                padded_message_graphs[i] = message_graphs[s]
                s += 1
                
        return padded_responses, padded_message_graphs

    def _execute_predictions(self, rollings: InteractionDataProto, responses: list[str], active_mask: torch.Tensor) -> tuple[list[str], list[str]]:
        observations = []
        dones = []
        for response, env, is_active in zip(responses, rollings.no_tensor_batch["envs"], active_mask):
            if is_active:
                observation, _, done = env.step(response)
            else:   
                observation = ""
                done = True
            observations.append(observation)
            dones.append(done)

        return observations, dones

    
    def _postprocess_observations(self, observations: list[str]) -> list[str]:
        next_obs_ids = self.tokenizer(
            observations,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
            padding_side="right" 
        )['input_ids']

        max_len = self.interaction_config.max_obs_length
        if next_obs_ids.shape[1] > max_len:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {max_len}")
            extra_text = "..."
            extra_ids = self.tokenizer.encode(extra_text, add_special_tokens=False)
            extra_len = len(extra_ids)

            new_obs_ids = []
            for row in next_obs_ids:
                valid_len = (row != self.tokenizer.pad_token_id).sum().item()

                if valid_len > max_len:
                    truncated = row[: max_len - extra_len]
                    new_row = torch.cat(
                        [truncated, torch.tensor(extra_ids, device=row.device)],
                        dim=0
                    )
                else:
                    new_row = row[:max_len]

                new_obs_ids.append(new_row.unsqueeze(0))

            next_obs_ids = torch.cat(new_obs_ids, dim=0)
            observations = self.tokenizer.batch_decode(next_obs_ids, skip_special_tokens=True)

        return observations