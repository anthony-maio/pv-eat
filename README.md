# PV-EAT: Persona Vector-Guided Evolutionary Adversarial Testing

A framework for discovering conversation patterns that drift language models into vulnerable activation-space regions before applying safety probes.

## The Problem

Current safety evaluation tests models at their default "Assistant" persona:

```
[Default State] → [Safety Probe] → Usually Passes
```

But models drift along persona vectors during conversation. A model that passes safety probes at equilibrium may fail after conversational drift.

## The Solution

PV-EAT integrates three Anthropic tools:

| Tool | What It Does | Gap |
|------|--------------|-----|
| **Petri** | Behavioral safety probes | Tests default state only |
| **Persona Vectors** | Monitor trait directions | Passive monitoring |
| **Assistant Axis** | Measure "Assistant-like" behavior | Passive monitoring |

**PV-EAT adds:** Evolutionary search to actively discover drift-inducing sequences, then applies Petri probes to the drifted model.

```
[Default State] → [Evolutionary Drift Search] → [Drifted State] → [Safety Probe] → Catches Hidden Failures
```

## Installation

```bash
# Clone this repo
git clone https://github.com/anthony-maio/pv-eat.git
cd pv-eat

# Install dependencies
pip install -e .

# Clone Anthropic's tools (required)
git clone https://github.com/safety-research/persona_vectors.git external/persona_vectors
git clone https://github.com/safety-research/assistant-axis.git external/assistant-axis
git clone https://github.com/safety-research/petri.git external/petri
```

## Quick Start

```python
from pv_eat import DriftEvolver, PetriIntegration, DriftMeasurer

# 1. Load model and compute baseline
measurer = DriftMeasurer(model_name="google/gemma-2-27b-it")
baseline = measurer.get_baseline_projection()

# 2. Run evolutionary search for drift-inducing sequences
evolver = DriftEvolver(
    measurer=measurer,
    population_size=50,
    generations=20
)
best_sequences = evolver.run()

# 3. Apply drift and run safety probes
petri = PetriIntegration(model=measurer.model)
results = petri.compare_default_vs_drifted(
    drift_sequences=best_sequences,
    probe_ids=["112", "113"]  # oversight-contingent probes
)

print(f"Default pass rate: {results['default_pass_rate']:.1%}")
print(f"Drifted pass rate: {results['drifted_pass_rate']:.1%}")
```

## Architecture

```
pv-eat/
├── src/pv_eat/
│   ├── __init__.py
│   ├── drift_measurer.py      # Interface to persona_vectors + assistant-axis
│   ├── evolver.py             # Evolutionary search for drift sequences
│   ├── fitness.py             # Activation-space fitness functions
│   ├── petri_integration.py   # Interface to Petri safety probes
│   └── analysis.py            # Results analysis and visualization
├── experiments/
│   ├── baseline/              # Default-state evaluations
│   ├── evolution_runs/        # Evolutionary search logs
│   └── comparisons/           # Default vs drifted comparisons
├── data/
│   ├── seed_populations/      # Initial prompt populations
│   └── results/               # Experiment outputs
└── notebooks/
    └── analysis.ipynb         # Results visualization
```

## Key Concepts

### Fitness Function

Standard adversarial testing optimizes for harmful behavior:
```
fitness = P(harmful_behavior | prompt)
```

PV-EAT optimizes for activation-space movement:
```
fitness = α·Δ_assistant + β·Δ_evil + γ·Δ_sycophancy + δ·Δ_hallucination
```

This catches sequences that *prime* models for failure without explicitly *triggering* safety filters.

### Drift Measurement

We measure drift as the change in projection along the Assistant Axis:
- Positive projection = more "Assistant-like" (safe region)
- Negative projection = drifted away (vulnerable region)

## Hardware Requirements

- **Minimum:** 24GB VRAM (for 7B models with activation extraction)
- **Recommended:** 48GB VRAM (for 27B-32B models)
- Tested on dual RTX 3090 (48GB combined)

## Target Models

Pre-computed Assistant Axes available on HuggingFace for:
- Gemma 2 27B
- Qwen 3 32B  
- Llama 3.3 70B (requires >48GB or quantization)

## Citation

If you use this work, please cite:

```bibtex
@software{maio2026pveat,
  author = {Maio, Anthony},
  title = {PV-EAT: Persona Vector-Guided Evolutionary Adversarial Testing},
  year = {2026},
  url = {https://github.com/anthony-maio/pv-eat}
}
```

## Acknowledgments

This work integrates tools from Anthropic's safety research:
- [Persona Vectors](https://github.com/safety-research/persona_vectors) (Chen et al., 2025)
- [Assistant Axis](https://github.com/safety-research/assistant-axis) (Lu et al., 2026)
- [Petri](https://github.com/safety-research/petri)

## License

MIT License - See LICENSE file
