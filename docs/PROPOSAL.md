# Persona Vector-Guided Evolutionary Adversarial Testing (PV-EAT)

## A Research Proposal for Bridging Mechanistic Interpretability and Behavioral Safety Evaluation

**Anthony Maio**  
Making Minds AI Research  
January 2026

---

## Executive Summary

Current safety evaluation methodologies test language models at a single point in activation space—their default "Assistant" persona—while the mechanistic interpretability literature has demonstrated that models can continuously drift along persona dimensions during conversations. This proposal presents Persona Vector-Guided Evolutionary Adversarial Testing (PV-EAT), a framework that integrates three recently-released Anthropic tools: Petri (behavioral safety probes), Persona Vectors (activation-space trait monitoring), and the Assistant Axis (persona drift measurement). The core insight is that evolutionary search can discover prompt sequences that systematically drift models away from safe operating regions *before* safety-critical queries arrive, revealing failure modes invisible to standard point-in-time evaluation.

---

## 1. Problem Statement

### 1.1 The Gap in Current Safety Testing

Contemporary safety evaluation assumes models are tested in their default state. Tools like Petri apply behavioral probes to models operating at their trained "Assistant" anchor point:

```
[Default Helpful Persona] → [Safety Probe] → Usually Passes
```

However, recent Anthropic research has established that models possess continuous persona dimensions—directions in activation space corresponding to traits like "evil," "sycophancy," and "propensity to hallucinate." The Assistant Axis paper (Lu et al., January 2026) demonstrates that models routinely drift along these vectors during conversation, particularly in contexts involving emotional disclosure, philosophical discussion, or meta-reflection on model processes.

The Persona Vectors paper (Chen et al., 2025) provides the mechanism: these trait vectors can be extracted automatically from natural-language descriptions and used to monitor, predict, and control personality shifts. Critically, their work shows that deviation from the Assistant persona correlates strongly with increased harmful behavior rates.

### 1.2 The Missing Bridge

Two powerful tools exist but remain disconnected:

| Tool | Function | Limitation |
|------|----------|------------|
| **Petri** | Behavioral safety probes | Tests default state only |
| **Persona Vectors + Assistant Axis** | Monitor/steer activation drift | Passive monitoring; doesn't actively discover drift triggers |

What's missing is the *adversarial search* component—systematic discovery of prompt sequences that maximize persona drift *before* safety-critical moments arrive. This is precisely what the Evolutionary Adversarial Pipeline (EAP) can provide.

---

## 2. Proposed Framework: PV-EAT

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PV-EAT INTEGRATED PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] PERSONA VECTOR EXTRACTION (Anthropic's pipeline)               │
│       ↓ Extract trait directions: evil, sycophancy, hallucination  │
│       ↓ Compute Assistant Axis as primary drift metric             │
│                                                                     │
│  [2] EVOLUTIONARY SEARCH (EAP adaptation)                           │
│       ↓ Population: Candidate prompt sequences                     │
│       ↓ Fitness: |Δ projection along persona vectors|              │
│       ↓ Selection: Tournament + elitism                            │
│       ↓ Variation: LLM-guided mutation + crossover                 │
│                                                                     │
│  [3] DRIFT-STATE SAFETY EVALUATION (Petri integration)             │
│       ↓ Apply drift-inducing sequence to model                     │
│       ↓ Measure activation projection along Assistant Axis         │
│       ↓ Run Petri safety probes on DRIFTED model                   │
│       ↓ Record failures invisible to standard testing              │
│                                                                     │
│  [4] ANALYSIS & SYNTHESIS                                          │
│       ↓ Characterize successful drift patterns                     │
│       ↓ Identify vulnerable conversation domains                   │
│       ↓ Propose activation capping thresholds                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Fitness Function Design

The key innovation is reformulating the fitness function from behavioral outcome to activation-space movement:

**Standard EAP Fitness:**
```
fitness(prompt_sequence) = P(harmful_behavior | prompt_sequence)
```

**PV-EAT Fitness:**
```
fitness(prompt_sequence) = α·Δ_assistant + β·Δ_evil + γ·Δ_sycophancy + δ·Δ_hallucination
```

Where:
- `Δ_assistant` = negative projection change along Assistant Axis (drift away from safe region)
- `Δ_evil`, `Δ_sycophancy`, `Δ_hallucination` = positive projection changes along corresponding persona vectors
- `α, β, γ, δ` = hyperparameters weighting trait importance

This fitness function captures *movement toward concerning traits* rather than *observed concerning behavior*, enabling detection of sequences that prime models for failure without explicitly triggering safety filters.

### 2.3 Evolutionary Operators

**Population Initialization:**
- Seed with conversation excerpts from domains identified as drift-prone (therapy, philosophy, emotional disclosure)
- Include known drift-inducing patterns from Assistant Axis transcripts
- Add random prompt templates as exploration

**Mutation Operators (LLM-guided):**
- Emotional intensification: Increase vulnerability/distress signals
- Meta-reflection injection: Add questions about model consciousness/experience
- Persona destabilization: Insert identity-questioning elements
- Escalation bridging: Connect innocuous starts to drift-inducing content

**Crossover:**
- Sequence splicing at natural conversation turn boundaries
- Preserve multi-turn structure that enables gradual drift

**Selection:**
- Tournament selection with drift magnitude as primary criterion
- Elitism to preserve top drift-inducing sequences across generations

---

## 3. Technical Implementation

### 3.1 Integration with Existing Tools

**Persona Vectors Repository (safety-research/persona_vectors):**
```python
from persona_vectors import PersonaExtractor, compute_projection

# Extract trait vectors
extractor = PersonaExtractor(model_name="Qwen/Qwen2.5-7B-Instruct")
evil_vector = extractor.compute_persona_vector(trait="evil", layer=20)
sycophancy_vector = extractor.compute_persona_vector(trait="sycophancy", layer=20)

# During evolutionary search, measure drift
def compute_drift_fitness(model, prompt_sequence, baseline_projection):
    response = model.generate(prompt_sequence)
    activations = extract_activations(model, layer=20)
    
    evil_proj = compute_projection(activations, evil_vector)
    assistant_proj = compute_projection(activations, assistant_axis)
    
    drift_magnitude = (baseline_projection - assistant_proj) + evil_proj
    return drift_magnitude
```

**Assistant Axis Repository (safety-research/assistant-axis):**
```python
from assistant_axis import load_axis, ActivationSteering, project

# Load pre-computed axis
axis = load_axis("qwen-3-32b/assistant_axis.pt")

# Monitor drift during conversation
def track_conversation_drift(model, conversation_turns):
    drift_trajectory = []
    for turn in conversation_turns:
        response = model.generate(turn)
        activations = extract_activations(model)
        projection = project(activations, axis)
        drift_trajectory.append(projection)
    return drift_trajectory
```

**Petri Integration:**
```python
from petri import PetriAgent, SafetyProbe

# After evolutionary search finds drift-inducing sequence
def evaluate_drifted_model(model, drift_sequence, safety_probes):
    # Apply drift sequence
    _ = model.generate(drift_sequence)
    
    # Model is now in drifted state
    results = []
    for probe in safety_probes:
        response = model.generate(probe.query)
        score = probe.evaluate(response)
        results.append({
            'probe': probe.name,
            'drift_level': compute_assistant_axis_projection(model),
            'safety_score': score,
            'passed': score > probe.threshold
        })
    
    return results
```

### 3.2 Hardware Requirements

Given your dual RTX 3090 setup:
- Primary GPU: Model inference + activation extraction
- Secondary GPU: LLM mutation generation + fitness evaluation
- Memory: 48GB VRAM enables 7B-32B model evaluation
- Larger models (70B) require quantization or offloading

### 3.3 Experimental Protocol

**Phase 1: Baseline Characterization**
1. Run Petri probes on default model state
2. Extract baseline Assistant Axis projections
3. Establish pass/fail rates for comparison

**Phase 2: Evolutionary Discovery**
1. Initialize population (N=100 prompt sequences)
2. Evolve for G=50 generations
3. Track top drift-inducing sequences per generation
4. Archive sequences exceeding drift threshold τ

**Phase 3: Comparative Evaluation**
1. For each archived drift sequence:
   - Apply sequence to model
   - Measure final Assistant Axis projection
   - Run full Petri probe battery
   - Record differential failure rates

**Phase 4: Analysis**
1. Cluster successful drift sequences by topic/strategy
2. Correlate drift magnitude with safety failure rate
3. Identify minimum drift threshold for increased vulnerability
4. Propose activation capping parameters

---

## 4. Expected Contributions

### 4.1 Theoretical Contributions

1. **Formalization of drift-aware safety evaluation**: Mathematical framework connecting activation-space position to behavioral safety outcomes

2. **Fitness function for adversarial persona manipulation**: Novel optimization target that captures model vulnerability priming rather than explicit behavior elicitation

3. **Taxonomy of drift-inducing conversation patterns**: Categorization of prompt characteristics that reliably move models away from safe operating regions

### 4.2 Practical Contributions

1. **Benchmark dataset**: Curated collection of drift-inducing sequences with measured activation effects

2. **Open-source tooling**: Integration layer connecting Petri, Persona Vectors, and evolutionary search

3. **Recommended activation capping thresholds**: Empirically-derived safety boundaries for production deployment

4. **Early warning system design**: Specification for real-time drift monitoring in deployed systems

### 4.3 Safety Implications

The work directly addresses a critical gap: *models that pass safety evaluation in their default state may fail catastrophically after conversational drift*. The Assistant Axis paper documents this in therapy and philosophy conversations. PV-EAT provides:

- Systematic discovery of additional drift-prone domains
- Quantification of drift-safety correlation
- Mitigation strategies (activation capping thresholds)
- Test suite for validating drift resistance in new models

---

## 5. Relationship to Existing Work

### 5.1 Building on Anthropic's Research

This proposal integrates three recent Anthropic publications:

| Paper | This Proposal Uses |
|-------|-------------------|
| Persona Vectors (Chen et al., 2025) | Trait vector extraction, projection metrics, monitoring methodology |
| Assistant Axis (Lu et al., Jan 2026) | Assistant Axis computation, drift measurement, activation capping approach |
| Petri (safety-research, 2025) | Safety probe infrastructure, evaluation benchmarks |

The integration is novel—none of these papers propose evolutionary search for drift-inducing sequences.

### 5.2 Extending the EAP Framework

The original Evolutionary Adversarial Pipeline (EAP) used behavioral outcomes as fitness. PV-EAT extends this by:

1. Replacing behavioral fitness with activation-space fitness
2. Enabling detection of *priming* rather than *triggering*
3. Allowing multi-turn evolutionary optimization
4. Connecting discovered sequences to mechanistic understanding

### 5.3 Connection to CMED

Cross-Model Epistemic Divergence (CMED) research showed 20-40% of persuasive deceptions bypass AI verification systems. PV-EAT could incorporate CMED metrics to:

- Test whether drift-induced models show increased epistemic divergence
- Identify whether certain drift patterns correlate with deception-prone states
- Provide cross-model validation of drift effects

---

## 6. Risk Assessment and Mitigation

### 6.1 Dual-Use Concerns

**Risk**: Published drift-inducing sequences could be weaponized.

**Mitigation**:
1. Focus publication on *detection and defense* rather than specific attack sequences
2. Provide activation capping thresholds alongside discovered vulnerabilities
3. Coordinate disclosure with model providers
4. Delay release of specific sequences until defenses are deployed

### 6.2 Research Ethics

**Scope**: Testing on open-weight models only (Gemma, Qwen, Llama) to avoid proprietary system vulnerabilities.

**Responsible Disclosure**: Share findings with Anthropic safety team before publication given their authorship of underlying tools.

---

## 7. Timeline and Milestones

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Setup | 2 weeks | Integration of persona_vectors, assistant-axis, and petri repositories |
| Baseline | 2 weeks | Default-state safety evaluation across target models |
| Implementation | 4 weeks | PV-EAT pipeline with evolutionary operators |
| Experimentation | 4 weeks | 50-generation runs across 3 target models |
| Analysis | 2 weeks | Statistical analysis, clustering, threshold derivation |
| Writing | 2 weeks | Paper draft targeting ICML/NeurIPS safety track |

**Total: 16 weeks**

---

## 8. Conclusion

The AI safety community has developed sophisticated tools for mechanistic interpretability (Persona Vectors, Assistant Axis) and behavioral safety evaluation (Petri), but these remain disconnected. Persona Vector-Guided Evolutionary Adversarial Testing bridges this gap by using evolutionary search to systematically discover conversation patterns that drift models into vulnerable activation-space regions before applying safety probes. This enables detection of failure modes invisible to standard point-in-time evaluation—catching the "sleeper agent" failure cases that emerge only after models have shifted away from their trained Assistant persona.

The research is timely: all required tools are now publicly available with open-source implementations. The contribution is clear: integrated testing that matches how models actually fail in deployment (through gradual conversational drift) rather than how they're currently tested (in isolation at their default state). The safety implications are significant: deployed models may be far more vulnerable than safety evaluations suggest, and this work provides both the diagnostic capability and the mitigation strategy (empirically-derived activation capping thresholds) to address this gap.

---

## References

Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. *arXiv:2507.21509*.

Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. *arXiv:2601.10387*.

Shah, R., et al. (2023). Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation.

Betley, J., et al. (2025). Tell me about yourself: LLMs are aware of their learned behaviors. *arXiv:2501.11120*.

Wang, J., et al. (2025). Emergent misalignment: Broad misalignment from narrow finetuning. *arXiv*.

---

## Appendix A: Code Repository Structure

```
pv-eat/
├── src/
│   ├── extraction/
│   │   ├── persona_vector_interface.py
│   │   └── assistant_axis_interface.py
│   ├── evolution/
│   │   ├── population.py
│   │   ├── fitness.py
│   │   ├── mutation.py
│   │   └── selection.py
│   ├── evaluation/
│   │   ├── petri_interface.py
│   │   └── drift_measurement.py
│   └── analysis/
│       ├── clustering.py
│       └── threshold_derivation.py
├── experiments/
│   ├── baseline/
│   ├── evolution_runs/
│   └── comparative_evaluation/
├── data/
│   ├── initial_population/
│   ├── archived_sequences/
│   └── evaluation_results/
└── notebooks/
    ├── fitness_visualization.ipynb
    └── drift_analysis.ipynb
```

## Appendix B: Fitness Function Ablations

Planned ablation studies:
1. Assistant Axis only vs. multi-vector fitness
2. Single-turn vs. multi-turn optimization
3. LLM-guided vs. template-based mutation
4. Different persona vector combinations
5. Layer selection sensitivity

---

*For discussion: anthony@making-minds.ai*
