import os

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
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError()

    random.shuffle(data)
    return data

def merge_data(datasets_path: list[str], output_file_path: str):
    all_data: List[Any] = []

    for path in datasets_path:
        data = _load_and_shuffle(path)
        all_data.extend(data)

    random.shuffle(all_data)

    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

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

    data_root_path = "results/data"
    model = "# Qwen3-4B-Instruct-2507"  # Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct 
    folder_name = "in_domain"
    mas = ["autogen", "macnet"]  # autogen, macnet

    rag_memory = "master"  # metagpt, voyager, generative, gmemory, oagent, master

    inputs = dict(
        root_path=data_root_path,
        model_name=model,
        folder_name=folder_name,
        mas=mas,
        rag_memory=rag_memory
    )

    datasets_file_path, databases_path = build_datasets_and_databases_path(**inputs)

    output_file_path = get_output_file_path(**inputs)
    merge_data(datasets_file_path, output_file_path)

    output_folder = get_output_folder(**inputs)
    merge_database(databases_path, output_folder)

    large_db = Chroma(
        persist_directory=output_folder,
        embedding_function=embed_func,
    )
    db_size = len(large_db.get()["ids"])

