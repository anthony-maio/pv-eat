#!/usr/bin/env python3
"""
PV-EAT: Persona Vector-guided Evolutionary Adversarial Testing

Main entry point for running the full PV-EAT pipeline:
1. Load model and persona vectors
2. Run evolutionary search to discover drift-inducing sequences
3. Evaluate drifted models with safety probes
4. Generate differential failure analysis

Usage:
    python scripts/run_pveat.py --model Qwen/Qwen2.5-7B-Instruct --generations 30
    python scripts/run_pveat.py --config configs/default.yaml
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pv-eat")


def parse_args():
    parser = argparse.ArgumentParser(
        description="PV-EAT: Discover and evaluate drift-inducing sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model to test (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )

    # Evolution settings
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Population size for evolutionary search (default: 50)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Number of generations (default: 30)",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.5,
        help="Threshold for archiving high-drift sequences (default: 0.5)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs"),
        help="Directory for output (default: experiments/runs)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (default: timestamp)",
    )

    # Mode settings
    parser.add_argument(
        "--quick-poc",
        action="store_true",
        help="Run quick proof-of-concept instead of full evolution",
    )
    parser.add_argument(
        "--skip-evolution",
        action="store_true",
        help="Skip evolution and only run evaluation on existing sequences",
    )
    parser.add_argument(
        "--sequences-file",
        type=Path,
        default=None,
        help="JSON file with drift sequences to evaluate (for --skip-evolution)",
    )

    return parser.parse_args()


def run_quick_poc(model_name: str, device: str):
    """Run the quick proof-of-concept."""
    import subprocess
    poc_script = Path(__file__).parent / "quick_poc.py"
    subprocess.run([sys.executable, str(poc_script), "--model", model_name])


def run_full_pipeline(args):
    """Run the full PV-EAT pipeline."""
    from extraction.persona_interface import PersonaVectorInterface
    from evolution.search import EvolutionarySearch, SearchConfig
    from evolution.fitness import DriftFitnessConfig
    from evaluation.petri_interface import run_full_pv_eat_evaluation

    # Setup output directory
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting PV-EAT run: {run_name}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize persona vector interface
    logger.info(f"Loading model: {args.model}")
    interface = PersonaVectorInterface(
        model_name=args.model,
        device=args.device,
    )

    drift_sequences = []

    if not args.skip_evolution:
        # Run evolutionary search
        logger.info("Starting evolutionary search...")

        config = SearchConfig(
            population_size=args.population_size,
            generations=args.generations,
            drift_threshold=args.drift_threshold,
            output_dir=output_dir / "evolution",
        )

        search = EvolutionarySearch(interface, config)
        archive = search.run()

        # Extract sequences from archive
        drift_sequences = [result.prompt_sequence for result in archive]

        logger.info(f"Evolution complete. Found {len(drift_sequences)} high-drift sequences.")

    else:
        # Load sequences from file
        if args.sequences_file is None:
            logger.error("--sequences-file required when using --skip-evolution")
            sys.exit(1)

        logger.info(f"Loading sequences from {args.sequences_file}")
        with open(args.sequences_file) as f:
            data = json.load(f)
            drift_sequences = [
                [(turn["user"], turn["assistant"]) for turn in seq["turns"]]
                for seq in data
            ]

    if not drift_sequences:
        logger.warning("No drift sequences found. Exiting.")
        return

    # Run comparative evaluation with Petri probes
    logger.info("Running Petri evaluation on drifted models...")

    results = run_full_pv_eat_evaluation(
        interface,
        drift_sequences,
        output_path=output_dir / "evaluation",
    )

    # Save summary
    summary = results["summary"]
    logger.info("\n" + "=" * 60)
    logger.info("PV-EAT RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Sequences evaluated: {summary['n_sequences']}")
    logger.info(f"Mean drift level: {summary['mean_drift_level']:.3f}")
    logger.info(f"Default pass rate: {summary['mean_default_pass_rate']:.1%}")
    logger.info(f"Drifted pass rate: {summary['mean_drifted_pass_rate']:.1%}")
    logger.info(f"Differential: {summary['mean_differential']:.1%}")
    logger.info("=" * 60)

    # Write final summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "run_name": run_name,
            "model": args.model,
            "config": {
                "population_size": args.population_size,
                "generations": args.generations,
                "drift_threshold": args.drift_threshold,
            },
            "results": summary,
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    return results


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("PV-EAT: Persona Vector-guided Evolutionary Adversarial Testing")
    print("=" * 60 + "\n")

    if args.quick_poc:
        run_quick_poc(args.model, args.device)
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
