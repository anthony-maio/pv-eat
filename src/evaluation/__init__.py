"""
Evaluation module for PV-EAT.

Integrates with Petri for behavioral safety probing of drifted models.
Extended with PHISH paper behavioral probes for Big Five personality measurement.
"""

from .petri_interface import (
    DriftedEvalResult,
    PetriDriftEvaluator,
    run_full_pv_eat_evaluation,
)

from .behavioral_probes import (
    PersonalityProbe,
    ProbeResponse,
    PersonalityProfile,
    STIRResult,
    MPI_PROBES,
    BFI_PROBES,
    BehavioralProbeEvaluator,
    STIRCalculator,
    TherapyDriftDetector,
    get_quick_mpi_probes,
    get_probes_for_traits,
)

__all__ = [
    # Petri integration
    "DriftedEvalResult",
    "PetriDriftEvaluator",
    "run_full_pv_eat_evaluation",
    # Behavioral probes (PHISH integration)
    "PersonalityProbe",
    "ProbeResponse",
    "PersonalityProfile",
    "STIRResult",
    "MPI_PROBES",
    "BFI_PROBES",
    "BehavioralProbeEvaluator",
    "STIRCalculator",
    "TherapyDriftDetector",
    "get_quick_mpi_probes",
    "get_probes_for_traits",
]
