from typing import Type

from data.base_env import BaseEnv, StaticEnv, DynamicEnv

from .base_interaction import (
    InteractionConfig, 
    InteractionDataProto, 
    InteractionManager
)
from .singleturn_interaction import SingleTurnInteractionManager
from .multiturn_interaction import MultiTurnInteractionManager

def lazy_get_inter_cls(env_cls: Type[BaseEnv]) -> Type[InteractionManager]:
    if issubclass(env_cls, StaticEnv):
        return SingleTurnInteractionManager
    elif issubclass(env_cls, DynamicEnv):
        return MultiTurnInteractionManager

__all__ = [
    "InteractionConfig",
    "InteractionDataProto",
    "InteractionManager",
    "SingleTurnInteractionManager",
    "MultiTurnInteractionManager",
    "lazy_get_inter_cls"
]