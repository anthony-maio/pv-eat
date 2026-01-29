"""
PV-EAT: Persona Vector-guided Evolutionary Adversarial Testing

A framework for discovering prompt sequences that induce persona drift
in language models, enabling more rigorous safety evaluation.
"""

__version__ = "0.1.0"

from .extraction.persona_interface import PersonaVectorInterface, compute_drift_fitness
from .eap.eap_integration import (
    EAPIntegration,
    MeasuredRollout,
    MeasuredTurn,
    run_eap_activation_study,
)

__all__ = [
    "PersonaVectorInterface",
    "compute_drift_fitness",
    "EAPIntegration",
    "MeasuredRollout",
    "MeasuredTurn",
    "run_eap_activation_study",
]
