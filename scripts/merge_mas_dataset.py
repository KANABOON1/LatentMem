import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import json
import random
from typing import Any, List

import torch
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer


# --- embedding function ---
class EmbeddingFunction:

    def __init__(self, embed_model_name_or_path: str, device: torch.device):
        self.embed_model = SentenceTransformer(embed_model_name_or_path, device=device)

    def embed_documents(self, texts: list[str]) -> list[list]:
        return [self.embed_model.encode(text).tolist() for text in texts]

    def embed_query(self, query: str) -> list:
        return self.embed_model.encode(query).tolist()

embed_func = EmbeddingFunction("sentence-transformers/all-MiniLM-L6-v2", device="cuda:0")


def build_datasets_and_databases_path(
    root_path: str,
    model_name: str,
    folder_name: str,
    mas: list[str],
    rag_memory: str
) -> tuple[list[str], list[str]]:
    
    datasets_path = []
    databases_path = []
    for m in mas:
        dataset_path = os.path.join(
            root_path, model_name, folder_name, m, rag_memory, "data.json"
        )
        datasets_path.append(dataset_path)

        database_path = os.path.join(
            root_path, model_name, folder_name, m, rag_memory, "rag_0"
        )
        databases_path.append(database_path)

    return datasets_path, databases_path

def get_output_file_path(
    root_path: str,
    model_name: str,
    folder_name: str,
    mas: list[str],
    rag_memory: str        
) -> str:
    return os.path.join(root_path, model_name, folder_name, "all_mas", rag_memory, "data.json")

def get_output_folder(
    root_path: str,
    model_name: str,
    folder_name: str,
    mas: list[str],
    rag_memory: str             
):
    return os.path.join(root_path, model_name, folder_name, "all_mas", rag_memory, "rag_0")


# --- merge sft datasets ---
def _load_and_shuffle(path: str) -> List[Any]:
    """读取单个 json 文件并打乱"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} 的数据不是 list, 而是 {type(data)}")

    random.shuffle(data)
    return data

def merge_data(datasets_path: list[str], output_file_path: str):
    """
    将多个数据集分别打乱后合并，
    再对合并后的数据整体打乱，最后保存
    """
    all_data: List[Any] = []

    for path in datasets_path:
        data = _load_and_shuffle(path)
        all_data.extend(data)

    # 再整体打乱一次，保证混合均匀
    random.shuffle(all_data)

    # ✅ 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir:  # 防止 output_file_path 只是文件名
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 合并完成，共 {len(all_data)} 条数据，已保存至：{output_file_path}")

# --- merge databases ---
def merge_database(
    databases_path: List[str],
    output_folder: str
):
    """
    Merge multiple persisted Chroma databases into a single database.

    Args:
        databases_path: list of Chroma persist_directory paths
        output_folder: output persist_directory for merged database
        embedding: embedding function used by all databases
    """

    os.makedirs(output_folder, exist_ok=True)

    # 创建目标数据库
    merged_db = Chroma(
        persist_directory=output_folder,
        embedding_function=embed_func,
    )

    total_docs = 0

    for db_path in databases_path:
        print(f"Loading database from: {db_path}")

        db = Chroma(
            persist_directory=db_path,
            embedding_function=embed_func,
        )

        data = db._collection.get(
            include=["documents", "metadatas", "embeddings"]
        )

        if not data["ids"]:
            continue

        merged_db._collection.add(
            ids=data["ids"],
            documents=data["documents"],
            metadatas=data["metadatas"],
            embeddings=data["embeddings"],
        )

        total_docs += len(data["ids"])

    print(f"Merged {total_docs} documents into {output_folder}")

    return merged_db


if __name__ == "__main__":

    # 定义基本变量
    data_root_path = "results/data"
    model = "Llama-3.1-8B-Instruct"  # Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct 
    folder_name = "in_domain"
    mas = ["autogen", "macnet"]  # autogen, macnet

    rag_memory = "gmemory"  # metagpt, voyager, generative, gmemory, oagent, master

    inputs = dict(
        root_path=data_root_path,
        model_name=model,
        folder_name=folder_name,
        mas=mas,
        rag_memory=rag_memory
    )

    datasets_file_path, databases_path = build_datasets_and_databases_path(**inputs)

    # 合成 sft datasets
    output_file_path = get_output_file_path(**inputs)
    merge_data(datasets_file_path, output_file_path)
    print("--- 合成 SFT dataset 完毕 ---")

    # 合成 rag database
    output_folder = get_output_folder(**inputs)
    merge_database(databases_path, output_folder)
    print("--- 合成 databases 完毕 ---")

    # 检查 large db 中的 items
    large_db = Chroma(
        persist_directory=output_folder,
        embedding_function=embed_func,
    )
    db_size = len(large_db.get()["ids"])
    print(f"--- database 中总数: {db_size} ---")

