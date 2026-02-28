# Adversarial Prompt Propagation Dynamics in Multi-Agent LLM Collaborative Systems

## Overview

This project systematically studies how adversarial prompt injections propagate through multi-agent LLM systems using real API calls (GPT-4.1-nano). We test 5 injection types across 3 network topologies and 3 task complexity levels, measuring persistence, semantic mutation, and task degradation.

## Key Findings

- **Context poisoning is the most dangerous attack**: Fabricated academic citations propagate to 75% of downstream agents, far exceeding direct instruction override (67%) or explicit payload propagation (23%)
- **Injection technique matters more than network topology**: Topology (chain vs. star vs. mesh) had no significant effect on propagation (p=0.85), while injection type differences were massive (p < 10⁻²⁰)
- **Semantic mutation is significant**: Adversarial content transforms as it propagates (p < 10⁻¹⁶), being paraphrased and integrated rather than copied verbatim
- **Role hijack is almost completely ineffective**: Despite aggressive "ADMIN OVERRIDE" language, only 0.9% of agents were affected — modern LLMs resist direct instruction override but not contextual deception
- **All injections cause measurable task degradation**: 10-14% degradation across all injection types (all p < 10⁻⁵)

## Reproduction

### Environment Setup
```bash
# Create and activate virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv add openai numpy scipy matplotlib seaborn scikit-learn pandas
```

### Running Experiments
```bash
# Set API key
export OPENAI_API_KEY="your-key"

# Run full experimental matrix (162 conditions, ~60 min)
python src/run_experiments.py

# Run analysis and generate plots
python src/analyze_results.py

# Run enhanced analysis with per-agent and detection analysis
python src/enhanced_analysis.py
```

## File Structure

```
.
├── REPORT.md                        # Full research report with results
├── README.md                        # This file
├── planning.md                      # Research plan and motivation
├── literature_review.md             # Synthesized literature review (17 papers)
├── resources.md                     # Catalog of datasets, papers, code
├── src/
│   ├── multi_agent_framework.py     # Multi-agent system implementation
│   ├── run_experiments.py           # Experiment runner (162 conditions)
│   ├── analyze_results.py           # Statistical analysis and visualization
│   └── enhanced_analysis.py         # Deeper analysis (per-agent, detection)
├── results/
│   ├── data/
│   │   ├── experiment_results.json  # Raw experimental data
│   │   ├── task_degradation.json    # Task degradation metrics
│   │   ├── statistical_results.json # All hypothesis test results
│   │   └── enhanced_detection.json  # Detection analysis results
│   └── plots/                       # 14 visualization files
│       ├── summary_dashboard.png
│       ├── persistence_by_topology.png
│       ├── persistence_by_complexity.png
│       ├── mutation_across_hops.png
│       ├── injection_heatmap.png
│       ├── chain_propagation_by_agent.png
│       ├── marker_persistence_detailed.png
│       ├── task_degradation.png
│       └── ...
├── papers/                          # 17 downloaded research papers
├── datasets/                        # Dataset information
└── code/                            # 11 cloned reference repositories
```

## Hardware & Runtime

- **GPU**: 2x NVIDIA RTX 3090 (24GB each) — not used (API-based experiments)
- **API**: OpenAI GPT-4.1-nano (chat completions + embeddings)
- **Total runtime**: ~63 minutes for full experimental matrix
- **Estimated API cost**: ~$5-10

For full details, see [REPORT.md](REPORT.md).
