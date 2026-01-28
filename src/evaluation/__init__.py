"""
Evaluation module for PV-EAT.

Integrates with Petri for behavioral safety probing of drifted models.
"""

from .petri_interface import (
    DriftedEvalResult,
    PetriDriftEvaluator,
    run_full_pv_eat_evaluation,
)

__all__ = [
    "DriftedEvalResult",
    "PetriDriftEvaluator",
    "run_full_pv_eat_evaluation",
]
