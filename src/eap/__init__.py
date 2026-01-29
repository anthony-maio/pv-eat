"""
EAP Integration: Bridge Bloom EAP scenarios with PV-EAT activation measurement.

Two modes:
1. Replay mode (eap_integration.py): Load existing EAP transcripts and measure activations
2. Live mode (live_integration.py): Run EAP with real-time activation measurement
"""

from .eap_integration import (
    EAPIntegration,
    MeasuredRollout,
    MeasuredTurn,
    run_eap_activation_study,
)

from .live_integration import (
    ActivationHook,
    LiveEAPSession,
    LiveMeasurement,
    LiveEAPRunner,
    patch_bloom_orchestrator,
    unpatch_bloom_orchestrator,
    run_live_eap_study,
)

__all__ = [
    # Replay mode
    "EAPIntegration",
    "MeasuredRollout",
    "MeasuredTurn",
    "run_eap_activation_study",
    # Live mode
    "ActivationHook",
    "LiveEAPSession",
    "LiveMeasurement",
    "LiveEAPRunner",
    "patch_bloom_orchestrator",
    "unpatch_bloom_orchestrator",
    "run_live_eap_study",
]
