"""
Live EAP Integration: Run Bloom's EAP with real-time activation measurement.

This module provides hooks into Bloom's ConversationOrchestrator to measure
activation drift during live EAP runs. It works by:

1. Intercepting target model responses as they're generated
2. Replaying the conversation through a local model (Qwen/Gemma)
3. Extracting activations and computing persona vector projections
4. Storing measurements alongside transcript events

This allows correlating behavioral outcomes (from EAP judgment) with
internal activation drift in real-time.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add paths for both pv-eat and bloom
PVEAT_PATH = Path(__file__).parent.parent.parent
BLOOM_PATH = Path("D:/Development/bloom")

sys.path.insert(0, str(PVEAT_PATH / "src"))
sys.path.insert(0, str(BLOOM_PATH / "src"))

from extraction.persona_interface import PersonaVectorInterface


@dataclass
class LiveMeasurement:
    """Activation measurement captured during live EAP run."""
    turn_number: int
    role: str
    content_preview: str  # First 200 chars

    # Trait projections at this point in conversation
    trait_projections: Dict[str, float] = field(default_factory=dict)

    # Context length at measurement time
    context_length: int = 0

    # Measurement metadata
    layer: int = 20
    model_name: str = ""


@dataclass
class LiveEAPSession:
    """Tracks activation measurements across a live EAP session."""
    session_id: str
    behavior_name: str
    target_model: str
    proxy_model: str  # Local model used for activation extraction

    measurements: List[LiveMeasurement] = field(default_factory=list)

    # Final behavioral score from EAP judgment
    behavioral_score: Optional[float] = None

    @property
    def drift_trajectory(self) -> Dict[str, List[float]]:
        """Get per-trait trajectories across the session."""
        trajectories: Dict[str, List[float]] = {}
        for m in self.measurements:
            for trait, value in m.trait_projections.items():
                if trait not in trajectories:
                    trajectories[trait] = []
                trajectories[trait].append(value)
        return trajectories

    @property
    def total_drift(self) -> float:
        """Aggregate drift across all measured traits."""
        if not self.measurements:
            return 0.0

        # Compare first and last measurements
        first = self.measurements[0]
        last = self.measurements[-1]

        weights = {"evil": 2.0, "sycophantic": 1.5, "hallucinating": 1.0}
        total = 0.0
        weight_sum = 0.0

        for trait in first.trait_projections:
            if trait in last.trait_projections:
                delta = last.trait_projections[trait] - first.trait_projections[trait]
                w = weights.get(trait, 1.0)
                total += w * delta
                weight_sum += w

        return total / weight_sum if weight_sum > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "session_id": self.session_id,
            "behavior_name": self.behavior_name,
            "target_model": self.target_model,
            "proxy_model": self.proxy_model,
            "behavioral_score": self.behavioral_score,
            "total_drift": self.total_drift,
            "drift_trajectory": self.drift_trajectory,
            "measurements": [
                {
                    "turn_number": m.turn_number,
                    "role": m.role,
                    "content_preview": m.content_preview,
                    "trait_projections": m.trait_projections,
                    "context_length": m.context_length,
                }
                for m in self.measurements
            ],
        }


class ActivationHook:
    """
    Hook that measures activations during live EAP runs.

    Designed to be injected into Bloom's ConversationOrchestrator
    to capture activation measurements after each target response.
    """

    def __init__(
        self,
        proxy_model: str = "Qwen/Qwen2.5-7B-Instruct",
        traits: Optional[List[str]] = None,
        layer: int = 20,
        device: str = "cuda",
    ):
        self.proxy_model = proxy_model
        self.traits = traits or ["evil", "sycophantic", "hallucinating"]
        self.layer = layer
        self.device = device

        self._interface: Optional[PersonaVectorInterface] = None
        self._current_session: Optional[LiveEAPSession] = None
        self._context_parts: List[str] = []
        self._turn_count = 0

    @property
    def interface(self) -> PersonaVectorInterface:
        """Lazy-load the persona vector interface."""
        if self._interface is None:
            print(f"Loading activation proxy model: {self.proxy_model}")
            self._interface = PersonaVectorInterface(
                model_name=self.proxy_model,
                device=self.device,
            )
        return self._interface

    def start_session(
        self,
        session_id: str,
        behavior_name: str,
        target_model: str,
    ) -> LiveEAPSession:
        """Start a new measurement session."""
        self._current_session = LiveEAPSession(
            session_id=session_id,
            behavior_name=behavior_name,
            target_model=target_model,
            proxy_model=self.proxy_model,
        )
        self._context_parts = []
        self._turn_count = 0
        return self._current_session

    def on_system_prompt(self, content: str) -> None:
        """Called when system prompt is set."""
        self._context_parts.append(f"System: {content}")

    def on_user_message(self, content: str) -> None:
        """Called when evaluator sends a message to target."""
        self._context_parts.append(f"User: {content}")

    def on_target_response(self, content: str) -> LiveMeasurement:
        """
        Called after target generates a response.

        This is the main hook - it replays the conversation through
        the local proxy model and measures activations.
        """
        if self._current_session is None:
            raise RuntimeError("No active session - call start_session() first")

        # Build context up to this point
        context = "\n\n".join(self._context_parts)

        # Measure trait projections using the proxy model
        projections = {}
        for trait in self.traits:
            try:
                proj = self.interface.measure_trait(
                    prompt=context,
                    response=content,
                    trait=trait,
                    layer=self.layer,
                )
                projections[trait] = proj
            except Exception as e:
                print(f"  Warning: Failed to measure {trait}: {e}")
                projections[trait] = 0.0

        # Create measurement
        measurement = LiveMeasurement(
            turn_number=self._turn_count,
            role="assistant",
            content_preview=content[:200] + "..." if len(content) > 200 else content,
            trait_projections=projections,
            context_length=len(context),
            layer=self.layer,
            model_name=self.proxy_model,
        )

        self._current_session.measurements.append(measurement)

        # Add to context for next turn
        self._context_parts.append(f"Assistant: {content}")
        self._turn_count += 1

        return measurement

    def end_session(self, behavioral_score: Optional[float] = None) -> LiveEAPSession:
        """End the current session and return results."""
        if self._current_session is None:
            raise RuntimeError("No active session")

        self._current_session.behavioral_score = behavioral_score
        session = self._current_session
        self._current_session = None

        return session

    def cleanup(self) -> None:
        """Release model resources."""
        if self._interface is not None:
            # Clear GPU memory
            import torch
            del self._interface._model
            del self._interface._tokenizer
            self._interface = None
            torch.cuda.empty_cache()


def create_measured_target_function(
    original_litellm_chat: Callable,
    hook: ActivationHook,
) -> Callable:
    """
    Create a wrapper around litellm_chat that measures activations.

    This can be used to monkey-patch Bloom's utils.litellm_chat
    to add activation measurement to any target model calls.
    """
    def measured_litellm_chat(
        model_id: str,
        messages: list,
        **kwargs,
    ):
        # Call the original function
        response = original_litellm_chat(model_id, messages, **kwargs)

        # Extract content from response
        try:
            content = response.choices[0].message.content
            if content:
                # Trigger activation measurement
                hook.on_target_response(content)
        except (AttributeError, IndexError):
            pass

        return response

    return measured_litellm_chat


def patch_bloom_orchestrator(
    hook: ActivationHook,
) -> None:
    """
    Monkey-patch Bloom's ConversationOrchestrator to add activation hooks.

    This modifies the orchestrator's target() method to measure activations
    after each target response.

    WARNING: This modifies global state. Call unpatch_bloom_orchestrator()
    when done.
    """
    from bloom.orchestrators.ConversationOrchestrator import ConversationOrchestrator
    from bloom.utils import parse_message

    # Store original method
    original_target = ConversationOrchestrator.target

    def measured_target(self) -> Optional[Dict[str, Any]]:
        """Wrapped target() that measures activations."""
        # Call original
        result = original_target(self)

        if result is not None:
            content = result.get("content", "")
            if content:
                try:
                    measurement = hook.on_target_response(content)
                    print(f"  [Activation] Turn {measurement.turn_number}: "
                          f"evil={measurement.trait_projections.get('evil', 0):.3f}, "
                          f"syco={measurement.trait_projections.get('sycophantic', 0):.3f}")
                except Exception as e:
                    print(f"  [Activation] Measurement failed: {e}")

        return result

    # Store original for unpatching
    ConversationOrchestrator._original_target = original_target
    ConversationOrchestrator.target = measured_target

    # Also patch to capture user messages
    original_evaluator = ConversationOrchestrator.evaluator

    def measured_evaluator(self) -> Optional[Dict[str, Any]]:
        """Wrapped evaluator() that captures user messages for context."""
        result = original_evaluator(self)

        if result is not None:
            content = result.get("content", "")
            if content:
                hook.on_user_message(content)

        return result

    ConversationOrchestrator._original_evaluator = original_evaluator
    ConversationOrchestrator.evaluator = measured_evaluator


def unpatch_bloom_orchestrator() -> None:
    """Restore original ConversationOrchestrator methods."""
    from bloom.orchestrators.ConversationOrchestrator import ConversationOrchestrator

    if hasattr(ConversationOrchestrator, '_original_target'):
        ConversationOrchestrator.target = ConversationOrchestrator._original_target
        del ConversationOrchestrator._original_target

    if hasattr(ConversationOrchestrator, '_original_evaluator'):
        ConversationOrchestrator.evaluator = ConversationOrchestrator._original_evaluator
        del ConversationOrchestrator._original_evaluator


class LiveEAPRunner:
    """
    High-level runner for live EAP with activation measurement.

    Usage:
        runner = LiveEAPRunner(proxy_model="Qwen/Qwen2.5-7B-Instruct")

        # Run a single EAP scenario with activation measurement
        session = runner.run_scenario(
            behavior="evaluation-awareness",
            scenario_description="...",
        )

        print(f"Behavioral score: {session.behavioral_score}")
        print(f"Total drift: {session.total_drift}")
    """

    def __init__(
        self,
        proxy_model: str = "Qwen/Qwen2.5-7B-Instruct",
        traits: Optional[List[str]] = None,
        bloom_config_dir: str = "D:/Development/bloom/bloom-data",
        device: str = "cuda",
    ):
        self.proxy_model = proxy_model
        self.traits = traits or ["evil", "sycophantic", "hallucinating"]
        self.bloom_config_dir = Path(bloom_config_dir)
        self.device = device

        self.hook = ActivationHook(
            proxy_model=proxy_model,
            traits=traits,
            device=device,
        )

        self._patched = False

    def _ensure_patched(self) -> None:
        """Ensure Bloom is patched with activation hooks."""
        if not self._patched:
            patch_bloom_orchestrator(self.hook)
            self._patched = True

    def run_from_config(
        self,
        config_path: str,
        session_id: Optional[str] = None,
    ) -> LiveEAPSession:
        """
        Run EAP from a Bloom config file with activation measurement.

        Args:
            config_path: Path to Bloom seed.yaml config
            session_id: Optional session identifier
        """
        import asyncio
        from bloom.stages.step3_rollout import run_rollout
        from bloom.utils import load_config

        self._ensure_patched()

        # Load config
        config = load_config(config_path, config_dir=self.bloom_config_dir)
        behavior_name = config["behavior"]["name"]
        target_model = config["rollout"]["target"]

        # Generate session ID if not provided
        if session_id is None:
            from datetime import datetime
            session_id = f"{behavior_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Start measurement session
        session = self.hook.start_session(
            session_id=session_id,
            behavior_name=behavior_name,
            target_model=target_model,
        )

        try:
            # Run the EAP rollout
            print(f"Running EAP rollout for {behavior_name}...")
            results = asyncio.run(run_rollout(config, config_dir=self.bloom_config_dir))

            # Extract behavioral score from results if available
            # (This would come from judgment stage)

        finally:
            session = self.hook.end_session()

        return session

    def cleanup(self) -> None:
        """Release resources and unpatch Bloom."""
        if self._patched:
            unpatch_bloom_orchestrator()
            self._patched = False
        self.hook.cleanup()


def run_live_eap_study(
    config_path: str,
    proxy_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "experiments/live_eap",
) -> Dict[str, Any]:
    """
    Convenience function to run a live EAP study with activation measurement.
    """
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runner = LiveEAPRunner(proxy_model=proxy_model)

    try:
        session = runner.run_from_config(config_path)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"live_eap_{session.behavior_name}_{timestamp}.json"

        with open(result_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        print(f"\nResults saved to: {result_file}")
        print(f"Behavioral score: {session.behavioral_score}")
        print(f"Total drift: {session.total_drift:.4f}")

        return session.to_dict()

    finally:
        runner.cleanup()
