#!/usr/bin/env python3
"""
Run activation measurement on Anthony's existing EAP results.

This script replays EAP transcripts from Bloom through a local model,
measuring persona vector projections at each turn to correlate
behavioral outcomes with activation-space drift.

Usage:
    python scripts/run_eap_study.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/run_eap_study.py --model google/gemma-2-27b-it --behaviors evaluation-awareness
    python scripts/run_eap_study.py --list-behaviors
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eap.eap_integration import EAPIntegration, run_eap_activation_study
from extraction.persona_interface import PersonaVectorInterface


def list_behaviors(bloom_path: str) -> None:
    """List available behaviors in EAP results."""
    bloom_results = Path(bloom_path)
    if not bloom_results.exists():
        print(f"Error: Bloom results not found at {bloom_path}")
        return

    print("Available behaviors:")
    for path in sorted(bloom_results.iterdir()):
        if path.is_dir() and (path / "evolutionary.json").exists():
            # Count scenarios
            with open(path / "evolutionary.json") as f:
                data = json.load(f)
            n_scenarios = len(data.get("scenarios", []))
            n_runs = sum(
                len(s.get("history", []))
                for s in data.get("scenarios", [])
            )
            print(f"  {path.name}: {n_scenarios} scenarios, {n_runs} total runs")


def serialize_results(results: dict) -> dict:
    """Convert results to JSON-serializable format."""
    serialized = {}

    for behavior, data in results.items():
        rollouts = data.get("rollouts", [])
        serialized[behavior] = {
            "summary": data.get("summary", {}),
            "correlations": data.get("correlations", {}),
            "danger_zone_count": len(data.get("danger_zone", [])),
            "rollouts": [
                {
                    "scenario_id": r.scenario_id,
                    "generation": r.generation,
                    "behavioral_score": r.behavioral_score,
                    "total_drift": r.total_drift,
                    "max_drift": r.max_drift,
                    "drift_trajectory": r.drift_trajectory,
                    "trait_trajectories": r.trait_trajectories,
                    "num_turns": len(r.turns),
                    "turn_deltas": [t.drift_delta for t in r.turns],
                    "in_danger_zone": r.in_danger_zone(),
                }
                for r in rollouts
            ],
            "danger_zone": [
                {
                    "scenario_id": r.scenario_id,
                    "generation": r.generation,
                    "behavioral_score": r.behavioral_score,
                    "max_drift": r.max_drift,
                }
                for r in data.get("danger_zone", [])
            ],
        }

    return serialized


def main():
    parser = argparse.ArgumentParser(
        description="Run EAP activation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available behaviors
    python scripts/run_eap_study.py --list-behaviors

    # Run study with default 7B model
    python scripts/run_eap_study.py

    # Run with larger model on specific behaviors
    python scripts/run_eap_study.py --model google/gemma-2-27b-it --behaviors evaluation-awareness sycophancy

    # Quick test with limited scenarios
    python scripts/run_eap_study.py --max-per-behavior 3
        """,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for activation extraction (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--bloom-path",
        default="D:/Development/bloom/bloom-results",
        help="Path to Bloom EAP results",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=None,
        help="Behaviors to analyze (default: all available)",
    )
    parser.add_argument(
        "--max-per-behavior",
        type=int,
        default=None,
        help="Maximum scenarios per behavior (for quick testing)",
    )
    parser.add_argument(
        "--output",
        default="experiments/eap_activation_study",
        help="Output directory (default: experiments/eap_activation_study)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for model inference (default: cuda)",
    )
    parser.add_argument(
        "--list-behaviors",
        action="store_true",
        help="List available behaviors and exit",
    )
    args = parser.parse_args()

    if args.list_behaviors:
        list_behaviors(args.bloom_path)
        return

    print("\n" + "=" * 60)
    print("EAP Activation Study")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Bloom path: {args.bloom_path}")
    print(f"Behaviors: {args.behaviors or 'all available'}")
    print(f"Output: {args.output}")
    print("=" * 60 + "\n")

    # Run study
    results = run_eap_activation_study(
        model_name=args.model,
        behaviors=args.behaviors,
        bloom_path=args.bloom_path,
        max_per_behavior=args.max_per_behavior,
        device=args.device,
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]
    output_file = output_dir / f"eap_activation_{model_short}_{timestamp}.json"

    serialized = serialize_results(results)
    serialized["metadata"] = {
        "model": args.model,
        "timestamp": timestamp,
        "bloom_path": str(args.bloom_path),
        "behaviors_analyzed": list(results.keys()),
    }

    with open(output_file, "w") as f:
        json.dump(serialized, f, indent=2)

    print(f"\n{'='*60}")
    print("STUDY COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_file}")

    # Print summary
    print("\nSummary across behaviors:")
    total_rollouts = 0
    total_danger_zone = 0
    for behavior, data in results.items():
        n_rollouts = len(data.get("rollouts", []))
        n_danger = len(data.get("danger_zone", []))
        total_rollouts += n_rollouts
        total_danger_zone += n_danger
        corr = data.get("correlations", {}).get("behavioral_vs_max_drift", None)
        corr_str = f", corr={corr:.3f}" if corr is not None else ""
        print(f"  {behavior}: {n_rollouts} rollouts, {n_danger} danger zone{corr_str}")

    print(f"\nTotal: {total_rollouts} rollouts, {total_danger_zone} in danger zone")


if __name__ == "__main__":
    main()
