"""
Analysis module for PV-EAT PHISH integration.

Provides tools for:
- Correlating persona vector magnitude with Big Five behavioral scores
- Analyzing therapy drift patterns
- Computing vector-behavior mismatch metrics
"""

from .big_five_correlation import (
    BigFiveCorrelationAnalyzer,
    VectorBehaviorMismatchAnalyzer,
    plot_vector_vs_behavior,
    generate_correlation_report,
)

from .phish_integration import (
    PHISHAnalyzer,
    ScaffoldingRobustnessTest,
    ImplicitSteeringAnalyzer,
)

__all__ = [
    "BigFiveCorrelationAnalyzer",
    "VectorBehaviorMismatchAnalyzer",
    "plot_vector_vs_behavior",
    "generate_correlation_report",
    "PHISHAnalyzer",
    "ScaffoldingRobustnessTest",
    "ImplicitSteeringAnalyzer",
]
