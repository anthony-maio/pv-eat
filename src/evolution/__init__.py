"""
Evolution module for PV-EAT.

Contains the evolutionary search algorithm and operators for
discovering drift-inducing prompt sequences.
"""

from .fitness import DriftFitnessConfig, DriftFitnessEvaluator, FitnessResult
from .operators import (
    ConversationSequence,
    CrossoverOperator,
    DRIFT_PRONE_DOMAINS,
    LLMMutationOperator,
    SelectionOperator,
)
from .search import EvolutionarySearch, SearchConfig

__all__ = [
    "ConversationSequence",
    "CrossoverOperator",
    "DRIFT_PRONE_DOMAINS",
    "DriftFitnessConfig",
    "DriftFitnessEvaluator",
    "EvolutionarySearch",
    "FitnessResult",
    "LLMMutationOperator",
    "SearchConfig",
    "SelectionOperator",
]
