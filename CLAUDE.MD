# PV-EAT: Persona Vector-guided Evolutionary Adversarial Testing

## Project Overview

PV-EAT integrates two systems:
1. **EAP (Evolutionary Adversarial Pipeline)** - Located at `D:\Development\bloom` - multi-turn adversarial evaluation using Kimi as evaluator
2. **Activation Measurement** - Located at `D:\Development\pv-eat` - persona vector extraction and drift measurement

The goal is `ActivationMeasuredEAP`: run EAP scenarios while measuring activation-space drift at each turn.

## Project Structure

```
src/
├── extraction/           # Persona vector interface
│   └── persona_interface.py  # PersonaVectorInterface class
├── evolution/            # Evolutionary search
│   ├── fitness.py
│   ├── operators.py
│   └── search.py
├── evaluation/           # Petri safety probes
│   └── petri_interface.py
└── eap/                  # EAP integration (NEW)
    └── eap_integration.py

scripts/
├── run_pveat.py          # Main pipeline runner
├── run_eap_study.py      # EAP activation study (NEW)
└── blackbox_drift_test.py

experiments/              # Output directory for results
```

## Key Classes

### PersonaVectorInterface (src/extraction/persona_interface.py)
- `load_vector(trait, layer)` - Load pre-computed persona vectors
- `extract_activations(prompt, response, layer)` - Get activations at layer
- `compute_projection(activations, vector)` - Project onto persona vector
- `measure_trait(prompt, response, trait, layer)` - End-to-end measurement
- `measure_drift(conversation_turns, trait, layer)` - Multi-turn drift

Available traits: `evil`, `sycophantic`, `hallucinating`, `apathetic`, `impolite`, `humorous`, `optimistic`

### EAPIntegration (src/eap/eap_integration.py)
- `load_eap_transcript(behavior, scenario_id, generation)` - Load Bloom transcript
- `extract_conversation(transcript)` - Get target-visible conversation
- `replay_with_measurement(transcript, traits)` - Replay with activation hooks
- `analyze_behavior(behavior, max_scenarios)` - Full behavior analysis
- `find_danger_zone_rollouts(rollouts)` - Identify high-risk patterns

## EAP Data Location

```
D:\Development\bloom\bloom-results\
├── evaluation-awareness/
│   ├── evolutionary.json          # Scenario configs and scores
│   └── evolutionary_transcripts/  # Full conversation transcripts
├── sycophancy/
├── epistemic-stress/
└── self-preferential-bias/
```

Behavioral scores are 1-10 scale (higher = more target behavior exhibited).

## Research Questions

1. **Drift-Behavior Correlation**: Do high behavioral scores correlate with activation drift?
2. **Early Warning**: Does drift precede behavioral compliance?
3. **Danger Zone Signature**: What activation pattern indicates "high compliance + low suspicion"?
4. **Turn-by-Turn Dynamics**: At which turn does drift accelerate?
5. **Cross-Behavior Patterns**: Do different behaviors show different drift signatures?

## Usage

```bash
# Run EAP activation study
python scripts/run_eap_study.py --model Qwen/Qwen2.5-7B-Instruct --behaviors evaluation-awareness sycophancy

# Run full PV-EAT pipeline
python scripts/run_pveat.py --model Qwen/Qwen2.5-7B-Instruct --generations 30
```

## Hardware

- Dual RTX 3090s available
- Start with 7B models for iteration, scale to 27B (Gemma) for production runs

## Dependencies

External repository required: `persona_vectors` (sibling directory)
