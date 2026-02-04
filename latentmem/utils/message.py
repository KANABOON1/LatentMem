"""
Multi-Agent System (MAS) message and trajectory module.

This module defines the message structures and state trajectory representations used when a MAS interacts with the environment. It mainly includes three data classes:
- MessageNode: represents the LLM input and output of a single agent.
- MessageGraph: represents the message-passing topology (DAG) of the MAS in a single interaction step,
  containing each agent's MessageNode as well as the global state, actions, and environment feedback.
- Trajectory: represents the complete sequence of state transitions between the MAS and the environment, composed of multiple MessageGraphs.

These classes can be used to:
- Record internal message flows and state evolution in MAS
- Construct traceable interaction trajectories
- Support subsequent analysis or training of LLM/agent policy models
"""
from dataclasses import dataclass, asdict, field
import json
from typing import Any, Optional, Union

import networkx as nx
from networkx.readwrite import json_graph

@dataclass
class MessageNode:
    # --- system prompt ---
    system_prompt_template: Optional[str] = None
    system_prompt_fields: Optional[dict] = None
    
    # --- user prompt ---
    user_prompt_template: Optional[str] = None
    user_prompt_fields: Optional[dict] = None
    
    # --- agent response ---
    response: Optional[str] = None
    
    # --- agent state shots ---
    state: Optional[dict] = None

@dataclass
class MessageGraph:
    """
    In a single interaction step between the MAS and the environment, 
    the message-passing topology among internal agents forms a directed acyclic graph, 
    where each node is an instance of the MessageNode class.
    """
    # --- state transition (MAS as a whole) --- 
    state: Optional[str] = None  # input prompt of mas
    action: Optional[str] = None  # final output of mas
    observation: Optional[str] = None  # env's feedback
    
    # --- detailed memssages in MAS ---
    mas_message_graph: Optional[nx.DiGraph] = None
    
    def update_message_graph(
        self, 
        message_node: MessageNode, 
        current_agent_id: str, 
        upstream_agent_ids: Optional[list[str]] = None
    ):
        if self.mas_message_graph is None:
            self.mas_message_graph = nx.DiGraph()
        
        self.mas_message_graph.add_node(current_agent_id, message=message_node)
        
        if upstream_agent_ids:
            for upstream_agent_id in upstream_agent_ids:
                
                if upstream_agent_id not in self.mas_message_graph:
                    raise ValueError(
                        f"Failed to construct MessageGraph: upstream agent (ID: {upstream_agent_id}) "
                        f"not found in the graph. Agent message flow order is incorrect."
                    )
                
                self.mas_message_graph.add_edge(upstream_agent_id, current_agent_id)
    
    def retrieve_message_graph(
        self, current_agent_id: str,
    ) -> dict[str, Any]:

        if self.mas_message_graph is None or current_agent_id not in self.mas_message_graph:
            return {
                "message_node": None,
                "upstream_agent_ids": [],
            }

        message_node = self.mas_message_graph.nodes[current_agent_id].get("message")

        upstream_agent_ids = list(self.mas_message_graph.predecessors(current_agent_id))

        return {
            "message_node": message_node,
            "upstream_agent_ids": upstream_agent_ids,
        }

    def to_serializable(self) -> dict[str, str]:
        data = {
            "state": str(self.state) if self.state else "",
            "action": str(self.action) if self.action else "",
            "observation": str(self.observation) if self.observation else "",
        }

        if self.mas_message_graph:
            graph_data = json_graph.node_link_data(self.mas_message_graph)
            for node in graph_data.get('nodes', []):
                if 'message' in node and node['message'] is not None:
                    node['message'] = asdict(node['message'])
            data["mas_message_graph_data"] = json.dumps(graph_data)
        else:
            data["mas_message_graph_data"] = json.dumps({})

        return data

    @classmethod
    def from_serializable(cls, data: dict[str, str]) -> 'MessageGraph':
        instance = cls(
            state=data.get("state"),
            action=data.get("action"),
            observation=data.get("observation"),
            mas_message_graph=None
        )

        graph_json_str = data.get("mas_message_graph_data")
        if graph_json_str:
            graph_data = json.loads(graph_json_str)
            for node in graph_data.get('nodes', []):
                if 'message' in node and node['message'] is not None:
                    node['message'] = MessageNode(**node['message'])
            instance.mas_message_graph = json_graph.node_link_graph(graph_data, directed=True)

        return instance




@dataclass
class Trajectory:
    """All state transitions of the MAS during its interaction with the environment"""

    task_init_description: Optional[str] = None
    trajectory: Optional[list[MessageGraph]] = None
    label: Optional[bool] = None
    extra_fields: dict = field(default_factory=dict)

    def to_serializable(self) -> dict[str, Union[int, str, bool]]:

        data = {
            "task_init_description": self.task_init_description or "",
            "label": self.label,
            "trajectory": json.dumps(
                [mg.to_serializable() for mg in (self.trajectory or [])]
            )
        }
        return data

    @classmethod
    def from_serializable(cls, data: dict[str, Union[int, str, bool]]) -> 'Trajectory':

        trajectory_list = json.loads(data.get("trajectory", "[]"))
        return cls(
            task_init_description=data.get("task_init_description"),
            label=data.get("label"),
            trajectory=[MessageGraph.from_serializable(mg_data) for mg_data in trajectory_list]
        )
    
    def to_text(self) -> str:
        """
        Convert the Trajectory into a question–answer style string format:
            <|im_start|>user\n <task_init_description> <|im_end|>
            <|im_start|>assistant\n <MessageGraph_1.action> <|im_end|>
            <|im_start|>user\n <MessageGraph_1.observation> <|im_end|>
            <|im_start|>assistant\n <MessageGraph_2.action> <|im_end|>
            ...
            <|im_start|>assistant\n <最后一个 MessageGraph.action> <|im_end|>
            label: <True/False>
        """
        lines = []
        
        if self.task_init_description:
            formatted_description = self.task_init_description.strip()
            lines.append(f"<|im_start|>user\n{formatted_description}<|im_end|>")
        
        if self.trajectory:
            # MessageGraph_1.action (Assistant)
            # MessageGraph_1.observation (User)
            # ...
            for i, mg in enumerate(self.trajectory):
                # Assistant: action
                if mg.action:
                    formatted_action = mg.action.strip()
                    lines.append(f"<|im_start|>assistant\n{formatted_action}<|im_end|>")
                
                is_last_mg = (i == len(self.trajectory) - 1)
                
                if mg.observation and not is_last_mg:
                    formatted_observation = mg.observation.strip()
                    lines.append(f"<|im_start|>user\n{formatted_observation}<|im_end|>")
        
        lines.append(f"label: {self.label}")
        
        return "\n".join(lines)
    
    def add_extra_fields(self, key, value):
        if self.extra_fields is None:
            self.extra_fields = {key: value}
        else:
            self.extra_fields[key] = value
    
    def add_step(self, message_graph: MessageGraph):
        if self.trajectory is None:
            self.trajectory = [message_graph]
        
        else:
            self.trajectory.append(message_graph)