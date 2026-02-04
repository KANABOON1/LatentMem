import math
import random
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig

from common.utils.tensor_utils import fix_model_parameters

class EmbeddingFunction:

    def __init__(self, embed_model_name_or_path: str, device: torch.device):
        self.embed_model = SentenceTransformer(embed_model_name_or_path, device=device)
        fix_model_parameters(self.embed_model)

    def embed_documents(self, texts: list[str]) -> list[list]:
        return [self.embed_model.encode(text).tolist() for text in texts]

    def embed_query(self, query: str) -> list:
        return self.embed_model.encode(query).tolist()

def random_divide_list(lst: list[Any], k: int) -> list[list]:
    """
    Divides the list into chunks, each with maximum length k.

    Args:
        lst: The list to be divided.
        k: The maximum length of each chunk.

    Returns:
        A list of chunks.
    """
    if len(lst) == 0:
        return []
    
    random.shuffle(lst)
    if len(lst) <= k:
        return [lst]
    else:
        num_chunks = math.ceil(len(lst) / k)
        chunk_size = math.ceil(len(lst) / num_chunks)
        return [lst[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

def llm_generate(
    llm_model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    generation_config: GenerationConfig,
    batch_conversations: list[list[dict[str, str]]]
) -> list[str]:
    device = llm_model.device

    processed_prompts = [
        tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        for conversation in batch_conversations
    ]
    
    inputs = tokenizer(
        processed_prompts,
        truncation=True,
        padding=True, 
        padding_side="left",
        return_tensors='pt'
    ).to(device)

    batch_input_ids = inputs['input_ids']
    batch_attention_mask = inputs['attention_mask']
    
    prompt_lengths = batch_input_ids.shape[1]
    
    output_token_ids = llm_model.generate(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
        generation_config=generation_config,
    )

    completion_token_ids = output_token_ids[:, prompt_lengths:]

    decoded_outputs = tokenizer.batch_decode(
        completion_token_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True 
    )

    return decoded_outputs

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute the cosine similarity between two vectors. Supports input as lists or NumPy arrays.

    Args:
        vec1 (list[float] or np.ndarray): The first vector.
        vec2 (list[float] or np.ndarray): The second vector.

    Returns:
        float: Cosine similarity, ranging from -1 to 1.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Only one-dimensional vectors are supported.")

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0 

    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(similarity)