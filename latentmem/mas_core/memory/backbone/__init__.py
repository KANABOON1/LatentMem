from typing import Type

from .base_rag import BaseRAGMemory, RAGMemoryResponse


def get_rag_cls(rag_mode: str) -> Type[BaseRAGMemory]:
    
    if rag_mode == "metagpt":
        from .metagpt import MetaGPT
        return MetaGPT
    elif rag_mode == "voyager":
        from .voyager import Voyager
        return Voyager
    elif rag_mode == "generative":
        from .generative import Generative
        return Generative
    elif rag_mode == "gmemory":
        from .gmemory import GMemory
        return GMemory
    elif rag_mode == "oagent":
        from .oagent import OAgentMemory
        return OAgentMemory
    elif rag_mode == "latentmem":
        from .experience import ExperienceBank
        return ExperienceBank
    else:
        raise ValueError(f"Cannot find corresponding rag mode: {rag_mode}.")
    
__all__ = [
    "BaseRAGMemory",
    "get_rag_cls",
    "RAGMemoryResponse"
]
