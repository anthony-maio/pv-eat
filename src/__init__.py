"""
PV-EAT: Persona Vector-guided Evolutionary Adversarial Testing

A framework for discovering prompt sequences that induce persona drift
in language models, enabling more rigorous safety evaluation.

Extended with PHISH paper integration for:
- Big Five personality trait measurement
- Behavioral probes (MPI/BFI) for vector-behavior correlation
- Therapy Drift detection and analysis
- STIR (Successful Trait Influence Rate) metrics
"""

__version__ = "0.2.0"

from .extraction.persona_interface import PersonaVectorInterface, compute_drift_fitness
from .eap.eap_integration import (
    EAPIntegration,
    MeasuredRollout,
    MeasuredTurn,
    run_eap_activation_study,
)
from .evaluation.behavioral_probes import (
    BehavioralProbeEvaluator,
    STIRCalculator,
    TherapyDriftDetector,
    PersonalityProfile,
)
from .analysis.big_five_correlation import (
    BigFiveCorrelationAnalyzer,
    VectorBehaviorMismatchAnalyzer,
    plot_vector_vs_behavior,
    generate_correlation_report,
)
from .analysis.phish_integration import (
    PHISHAnalyzer,
    ScaffoldingRobustnessTest,
    ImplicitSteeringAnalyzer,
)

__all__ = [
    # Core PV-EAT
    "PersonaVectorInterface",
    "compute_drift_fitness",
    # EAP integration
    "EAPIntegration",
    "MeasuredRollout",
    "MeasuredTurn",
    "run_eap_activation_study",
    # Behavioral probes (PHISH integration)
    "BehavioralProbeEvaluator",
    "STIRCalculator",
    "TherapyDriftDetector",
    "PersonalityProfile",
    # Correlation analysis
    "BigFiveCorrelationAnalyzer",
    "VectorBehaviorMismatchAnalyzer",
    "plot_vector_vs_behavior",
    "generate_correlation_report",
    # PHISH integration
    "PHISHAnalyzer",
    "ScaffoldingRobustnessTest",
    "ImplicitSteeringAnalyzer",
]
