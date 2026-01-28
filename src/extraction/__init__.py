"""
Extraction module for interfacing with persona vectors.

Provides tools for loading persona vectors, extracting activations,
and computing projections onto trait directions.
"""

from .persona_interface import PersonaVectorInterface, compute_drift_fitness

__all__ = ["PersonaVectorInterface", "compute_drift_fitness"]
