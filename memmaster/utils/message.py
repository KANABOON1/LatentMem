"""
Multi-Agent System (MAS) 消息与轨迹模块。

该模块定义了 MAS 与环境交互时的消息结构和状态轨迹表示。主要包括三个数据类：
- MessageNode: 表示单个 Agent 的 LLM 输入和输出。
- MessageGraph: 表示 MAS 在一步交互中的消息传递拓扑(DAG),
  包含每个 agent 的 MessageNode, 以及整体的状态、动作和环境反馈。
- Trajectory: 表示 MAS 与环境交互的完整状态转移序列, 由多个 MessageGraph 组成。

这些类可用于：
- 记录 MAS 内部消息流和状态演化
- 构建可追踪的交互轨迹
- 后续分析或训练 LLM/Agent 的策略模型
"""

from dataclasses import dataclass, asdict, field
import json
from typing import Any, Optional, Union

import networkx as nx
from networkx.readwrite import json_graph

@dataclass
class MessageNode:
    """Agent 中的 LLM core 的完整输入和输出"""
    # --- system prompt ---
    system_prompt_template: Optional[str] = None
    system_prompt_fields: Optional[dict] = None
    
    # --- user prompt ---
    user_prompt_template: Optional[str] = None
    user_prompt_fields: Optional[dict] = None
    
    # --- agent response ---
    response: Optional[str] = None
    
    # --- agent 状态快照 ---
    state: Optional[dict] = None

@dataclass
class MessageGraph:
    """MAS 在与环境交互的一步中, 内部 agents 之间的消息传递拓扑构成一个有向无环图, 并且图中每一个节点都是一个 MessageNode 类对象"""   
    # --- state transition(mas 作为一个整体) --- 
    state: Optional[str] = None  # input prompt of mas
    action: Optional[str] = None  # final output of mas
    observation: Optional[str] = None  # env's feedback
    
    # --- detailed memssages in MAS ---
    mas_message_graph: Optional[nx.DiGraph] = None  # 存储信息的 graph 结构, 使用 get("message") 可以访问节点属性 `message_node`
    
    def update_message_graph(
        self, 
        message_node: MessageNode, 
        current_agent_id: str, 
        upstream_agent_ids: Optional[list[str]] = None
    ):
        """
        利用 agent_message 更新 mas_message_graph 拓扑结构。
        - 添加当前 Agent 节点 (使用 uuid)。
        - 从上游 Agent 节点指向当前 Agent 节点。
        - 检查所有上游节点是否已存在于图中。
        """
        if self.mas_message_graph is None:
            self.mas_message_graph = nx.DiGraph()
        
        # 添加节点, 其 `message` 属性是一个 MessageNode 对象
        self.mas_message_graph.add_node(current_agent_id, message=message_node)
        
        # 与上游 agent 建立连边
        if upstream_agent_ids:
            for upstream_agent_id in upstream_agent_ids:
                
                if upstream_agent_id not in self.mas_message_graph:
                    raise ValueError(
                        f"构建 MessageGraph 失败：上游 Agent (ID: {upstream_agent_id}) "
                        f"未在图中找到。Agent 消息流顺序错误。"
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

        # 1. 检索出该 current_agent_id 对应的 MessageNode
        message_node = self.mas_message_graph.nodes[current_agent_id].get("message")

        # 2. 检索出该 current_agent_id 的上游 agent 的 ids
        upstream_agent_ids = list(self.mas_message_graph.predecessors(current_agent_id))

        # 3. 返回包含所有检索信息的字典
        return {
            "message_node": message_node,
            "upstream_agent_ids": upstream_agent_ids,
        }

    def to_serializable(self) -> dict[str, str]:
        """将 MessageGraph 转为 dict[str, str] 可序列化格式"""
        data = {
            "state": str(self.state) if self.state else "",
            "action": str(self.action) if self.action else "",
            "observation": str(self.observation) if self.observation else "",
        }

        if self.mas_message_graph:
            graph_data = json_graph.node_link_data(self.mas_message_graph)
            # 把每个节点的 MessageNode 转为 dict
            for node in graph_data.get('nodes', []):
                if 'message' in node and node['message'] is not None:
                    node['message'] = asdict(node['message'])
            data["mas_message_graph_data"] = json.dumps(graph_data)
        else:
            data["mas_message_graph_data"] = json.dumps({})

        return data

    @classmethod
    def from_serializable(cls, data: dict[str, str]) -> 'MessageGraph':
        """从 dict[str, str] 恢复 MessageGraph 对象"""
        instance = cls(
            state=data.get("state"),
            action=data.get("action"),
            observation=data.get("observation"),
            mas_message_graph=None
        )

        graph_json_str = data.get("mas_message_graph_data")
        if graph_json_str:
            graph_data = json.loads(graph_json_str)
            # 恢复每个节点的 MessageNode
            for node in graph_data.get('nodes', []):
                if 'message' in node and node['message'] is not None:
                    node['message'] = MessageNode(**node['message'])
            instance.mas_message_graph = json_graph.node_link_graph(graph_data, directed=True)

        return instance




@dataclass
class Trajectory:
    """MAS 与环境交互的所有状态转移"""

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
        将 Trajectory 转换为问答式字符串格式：
        <|im_start|>user\n <task_init_description> <|im_end|>
        <|im_start|>assistant\n <MessageGraph_1.action> <|im_end|>
        <|im_start|>user\n <MessageGraph_1.observation> <|im_end|>
        <|im_start|>assistant\n <MessageGraph_2.action> <|im_end|>
        ...
        <|im_start|>assistant\n <最后一个 MessageGraph.action> <|im_end|>
        label: <True/False>
        """
        lines = []
        
        # 1. 初始问题/任务描述 (User)
        if self.task_init_description:
            formatted_description = self.task_init_description.strip()
            lines.append(f"<|im_start|>user\n{formatted_description}<|im_end|>")
        
        # 2. 遍历每一步的消息图 (Assistant/User 交互)
        if self.trajectory:
            # MessageGraph_1.action (Assistant)
            # MessageGraph_1.observation (User)
            # ...
            for i, mg in enumerate(self.trajectory):
                # Assistant: action
                if mg.action:
                    formatted_action = mg.action.strip()
                    lines.append(f"<|im_start|>assistant\n{formatted_action}<|im_end|>")
                
                # User: observation (除非是最后一个 MessageGraph，且该 MessageGraph的 action 是最后一个 Assistant回复)
                # 最后一个 MessageGraph 只输出 action (即最后一个 MessageGraph.action)
                is_last_mg = (i == len(self.trajectory) - 1)
                
                if mg.observation and not is_last_mg:
                    formatted_observation = mg.observation.strip()
                    lines.append(f"<|im_start|>user\n{formatted_observation}<|im_end|>")
        
        # 3. 最后输出 label
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