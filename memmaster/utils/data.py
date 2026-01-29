from dataclasses import asdict
import json
import os

from langchain_chroma import Chroma
from networkx.readwrite import json_graph

from memmaster.utils import message

def _rag_to_text_json(rag_dir):
    """
    从 Chroma RAG 存储目录中提取所有数据并导出为 JSON 文件。
    """
    json_file_path = os.path.join(rag_dir, "data.json")
    embedding_function = None

    # 加载已持久化的 Chroma 数据库
    main_memory = Chroma(
        embedding_function=embedding_function,
        persist_directory=rag_dir
    )

    try:
        # 获取所有数据
        data = main_memory.get(include=['documents', 'metadatas'])

        all_ids = data.get('ids', [])
        if not all_ids:
            print("The RAG memory (Chroma DB) is empty. Creating an empty JSON file.")
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            return

        documents = data.get('documents', [])
        metadatas = data.get('metadatas', [])

        # 构建样本结构
        all_samples = []
        for doc_content, meta_data in zip(documents, metadatas):
            sample_data = {
                "page_content": doc_content,
                "metadata": meta_data
            }
            all_samples.append(sample_data)

        # 写入 JSON 文件
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=4)

        print(f"✅ Successfully dumped {len(all_samples)} samples to {json_file_path}.")

    except Exception as e:
        print(f"❌ An error occurred during JSON dumping: {e}")

def _text_json_to_structure(json_path: str) -> list[message.Trajectory]:

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found")

    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected JSON file to contain a list of samples")

    recovered: list[message.Trajectory] = []
    failed_indices = []

    for idx, sample in enumerate(raw):
        try:
            # 标准样本格式： {"page_content": "...", "metadata": {...}}
            if isinstance(sample, dict):
                # 优先从 metadata 恢复（最常见的情况）
                metadata = sample.get("metadata")
                if isinstance(metadata, dict) and "trajectory" in metadata:
                    traj = message.Trajectory.from_serializable(metadata)
                    recovered.append(traj)
                    continue

                # 有时样本本身就是序列化的 Trajectory dict
                if "trajectory" in sample or "task_init_description" in sample:
                    # 直接用样本字典恢复（容错）
                    traj = message.Trajectory.from_serializable(sample)
                    recovered.append(traj)
                    continue

            # 如果到了这里仍未恢复，记录失败
            failed_indices.append(idx)
        except Exception as e:
            # 记录失败，但继续处理剩余样本
            failed_indices.append(idx)
            print(f"Warning: failed to parse sample index {idx}: {e}")

    print(f"Recovered {len(recovered)} trajectories; failed to recover {len(failed_indices)} samples.")
    if failed_indices:
        print(f"Failed indices (sample positions): {failed_indices}")

    return recovered

def _structure_to_text_json(recovered: list[message.Trajectory], output_path: str):
    """
    将 list[Trajectory] (recovered) 导出为结构化 JSON 文件，
    每一个字段层次清晰展开（不再嵌套为字符串）。
    """

    def message_node_to_dict(node: message.MessageNode) -> dict:
        """展开 MessageNode 为 dict"""
        return asdict(node)

    def message_graph_to_dict(graph: message.MessageGraph) -> dict:
        """展开 MessageGraph 为 dict"""
        if graph.mas_message_graph:
            graph_data = json_graph.node_link_data(graph.mas_message_graph)
            # 将每个节点的 MessageNode 对象转换为字典
            for node in graph_data.get('nodes', []):
                if 'message' in node and isinstance(node['message'], message.MessageNode):
                    node['message'] = message_node_to_dict(node['message'])
        else:
            graph_data = {}
        
        return {
            "state": graph.state,
            "action": graph.action,
            "observation": graph.observation,
            "mas_message_graph": graph_data
        }

    def trajectory_to_dict(traj: message.Trajectory) -> dict:
        """展开 Trajectory 为 dict"""
        return {
            "task_init_description": traj.task_init_description,
            "label": traj.label,
            "trajectory": [message_graph_to_dict(mg) for mg in (traj.trajectory or [])]
        }

    # 构造完整 JSON 结构
    json_data = [trajectory_to_dict(t) for t in recovered]

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Successfully saved {len(recovered)} trajectories to {output_path}")


def rag_to_structured_json(rag_dir: str):
    temp_json_path = os.path.join(rag_dir, "data_tmp.json")
    final_json_path = os.path.join(rag_dir, "data.json")

    print("🚀 [Step 1/4] Extracting raw text JSON from RAG...")
    _rag_to_text_json(rag_dir)

    raw_json_path = os.path.join(rag_dir, "data.json")
    if not os.path.exists(raw_json_path):
        raise FileNotFoundError(f"❌ data.json not found in {rag_dir}, extraction failed.")

    os.rename(raw_json_path, temp_json_path)

    print("🔄 [Step 2/4] Reconstructing trajectories from text JSON...")
    recovered = _text_json_to_structure(temp_json_path)

    print("💾 [Step 3/4] Saving structured trajectories to final JSON...")
    _structure_to_text_json(recovered, final_json_path)

    print("🧹 [Step 4/4] Cleaning up temporary files...")
    try:
        os.remove(temp_json_path)
        print(f"🗑️ Removed temporary file: {temp_json_path}")
    except Exception as e:
        print(f"⚠️ Warning: failed to remove temporary file ({temp_json_path}): {e}")

    print(f"✅ All done! Structured JSON saved to: {final_json_path}")


def structured_json_to_trajectories(json_path: str) -> list[message.Trajectory]:
    """
    从结构化 JSON 文件恢复为 list[Trajectory]。
    这个 JSON 是由 `_structure_to_text_json` 输出的, 每个字段已展开。
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found")

    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected JSON file to contain a list of trajectories")

    recovered: list[message.Trajectory] = []
    failed_indices = []

    def dict_to_message_node(node_dict: dict) -> message.MessageNode:
        """从 dict 恢复 MessageNode"""
        return message.MessageNode(**node_dict)

    def dict_to_message_graph(graph_dict: dict) -> message.MessageGraph:
        """从 dict 恢复 MessageGraph"""
        mas_graph_data = graph_dict.get("mas_message_graph", {})
        graph_obj = message.MessageGraph(
            state=graph_dict.get("state"),
            action=graph_dict.get("action"),
            observation=graph_dict.get("observation"),
            mas_message_graph=None  # 先设置为 None，后面恢复节点
        )

        if mas_graph_data.get("nodes"):
            # 用 networkx.json_graph.node_link_graph 恢复原始图
            g = json_graph.node_link_graph(mas_graph_data)
            # 将节点的 message 字段从 dict 转回 MessageNode
            for n, data in g.nodes(data=True):
                if "message" in data and isinstance(data["message"], dict):
                    data["message"] = dict_to_message_node(data["message"])
            graph_obj.mas_message_graph = g

        return graph_obj

    for idx, traj_dict in enumerate(raw):
        try:
            trajectory_list = traj_dict.get("trajectory", [])
            traj_obj = message.Trajectory(
                task_init_description=traj_dict.get("task_init_description"),
                label=traj_dict.get("label"),
                trajectory=[dict_to_message_graph(mg) for mg in trajectory_list]
            )
            recovered.append(traj_obj)
        except Exception as e:
            failed_indices.append(idx)
            print(f"Warning: failed to parse trajectory index {idx}: {e}")

    print(f"Recovered {len(recovered)} trajectories; failed to recover {len(failed_indices)} samples.")
    if failed_indices:
        print(f"Failed indices (positions in JSON): {failed_indices}")

    return recovered
