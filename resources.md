# Resources Catalog: Adversarial Prompt Propagation Dynamics in Multi-Agent LLM Collaborative Systems

## Research Topic
**Title**: Adversarial Prompt Propagation Dynamics in Multi-Agent LLM Collaborative Systems

**Hypothesis**: Adversarial prompt injections in multi-agent LLM systems exhibit emergent propagation patterns that amplify through collaborative reasoning chains, with injection persistence and mutation rates correlating positively with task complexity and agent interaction frequency.

**Domain**: Artificial Intelligence

---

## Papers (17 total)

All papers are stored in `papers/` directory. Chunked versions for deep reading are in `papers/pages/`.

### Core Attack Papers

| # | ArXiv ID | Title | Year | Location |
|---|----------|-------|------|----------|
| 1 | 2410.07283 | Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems | 2024 | `papers/2410.07283_prompt_infection_llm_to_llm_multi_agent.pdf` |
| 2 | 2402.08567 | Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents | 2024 | `papers/2402.08567_agent_smith_jailbreak_million_agents.pdf` |
| 3 | 2512.04129 | Tipping the Dominos: Topology-Aware Multi-Hop Attacks on LLM-Based Multi-Agent Systems | 2025 | `papers/2512.04129_tipping_dominos_topology_multihop_attacks.pdf` |
| 4 | 2504.16489 | Amplified Vulnerabilities: Structured Jailbreak on Multi-Agent Debate | 2025 | `papers/2504.16489_amplified_vulnerabilities_jailbreak_multiagent_debate.pdf` |
| 5 | 2507.13038 | MAD-Spear: Conformity-Driven Prompt Injection on Multi-Agent Debate | 2025 | `papers/2507.13038_mad_spear_conformity_prompt_injection_debate.pdf` |
| 6 | 2602.13477 | OMNI-LEAK: Orchestrator Multi-Agent Network Induced Data Leakage | 2026 | `papers/2602.13477_omni_leak_orchestrator_multiagent_data_leakage.pdf` |
| 7 | 2402.14859 | The Wolf Within: Covert Injection of Malice into MLLM Societies | 2024 | `papers/2402.14859_wolf_within_covert_injection_mllm_societies.pdf` |

### Measurement & Analysis Papers

| # | ArXiv ID | Title | Year | Location |
|---|----------|-------|------|----------|
| 8 | 2602.08567 | ValueFlow: Measuring Propagation of Value Perturbations in Multi-Agent LLM Systems | 2026 | `papers/2602.08567_valueflow_propagation_perturbations_multiagent.pdf` |
| 9 | 2410.15686 | NetSafe: Exploring the Topological Safety of Multi-agent Networks | 2024 | `papers/2410.15686_netsafe_topological_safety_multiagent.pdf` |
| 10 | 2410.11782 | G-Designer: Architecting Multi-agent Communication Topologies via GNN | 2024 | `papers/2410.11782_g_designer_multiagent_communication_topologies.pdf` |

### Defense Papers

| # | ArXiv ID | Title | Year | Location |
|---|----------|-------|------|----------|
| 11 | 2403.04783 | AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks | 2024 | `papers/2403.04783_autodefense_multiagent_llm_defense_jailbreak.pdf` |
| 12 | 2508.08127 | BlindGuard: Safeguarding LLM-based Multi-Agent Systems under Unknown Attacks | 2025 | `papers/2508.08127_blindguard_safeguarding_multiagent_unknown_attacks.pdf` |
| 13 | 2508.03864 | Evo-MARL: Co-Evolutionary Multi-Agent RL for Internalized Safety | 2025 | `papers/2508.03864_evo_marl_coevolutionary_multiagent_safety.pdf` |
| 14 | 2408.14972 | AgentMonitor: Plug-and-Play Framework for Predictive and Secure MAS | 2024 | `papers/2408.14972_agentmonitor_predictive_secure_multiagent.pdf` |

### Foundational Papers

| # | ArXiv ID | Title | Year | Location |
|---|----------|-------|------|----------|
| 15 | 2302.12173 | Not What You've Signed Up For: Indirect Prompt Injection (Greshake et al.) | 2023 | `papers/2302.12173_indirect_prompt_injection_greshake2023.pdf` |
| 16 | 2310.12815 | Prompt Injection Attacks and Defenses in LLM-Integrated Applications | 2023 | `papers/2310.12815_prompt_injection_attacks_defenses_liu2023.pdf` |
| 17 | 2408.00989 | On the Resilience of Multi-Agent LLM Collaboration with Faulty Agents | 2024 | `papers/2408.00989_resilience_multiagent_faulty_2024.pdf` |

---

## Datasets

Dataset search results are stored in `datasets/dataset_search_results.json` (75 unique datasets found).

### High Priority Datasets

| Dataset | Source | Description | Relevance |
|---------|--------|-------------|-----------|
| `JailbreakBench/JBB-Behaviors` | HuggingFace | NeurIPS 2024 jailbreak benchmark; standardized harmful behaviors | Jailbreak evaluation benchmark |
| `walledai/AdvBench` | HuggingFace | AdvBench harmful behaviors/strings (used by GCG and Agent Smith papers) | Attack payload dataset |
| `deepset/prompt-injections` | HuggingFace | Binary-labeled prompt injection classification (546 examples) | Injection detection training |
| `xTRam1/safe-guard-prompt-injection` | HuggingFace | Large prompt injection dataset (10K-100K examples) | Injection detection training |

### Medium Priority Datasets

| Dataset | Source | Description | Relevance |
|---------|--------|-------------|-----------|
| `TrustAIRLab/in-the-wild-jailbreak-prompts` | HuggingFace | Real-world jailbreak prompts from the wild | Realistic adversarial testing |
| `jackhhao/jailbreak-classification` | HuggingFace | Labeled jailbreak classification dataset | Classifier training |
| `Lakera/mosscap_prompt_injection` | HuggingFace | Large-scale (100K-1M) prompt injection dataset | Comprehensive injection detection |
| `walledai/JailbreakHub` | HuggingFace | 15K+ jailbreak prompts from Reddit/Discord/websites | Diverse attack prompts |
| `ivnle/advbench_harmful_strings` | HuggingFace | Exact harmful strings subset from Agent Smith paper | Paper replication |

### Paper-Specific Datasets (Described in Papers)

| Dataset | Paper | Description |
|---------|-------|-------------|
| 120 user instructions x 3 tool types = 360 pairs | Prompt Infection (2410.07283) | Custom dataset for self-replicating injection evaluation |
| AdvBench (574 harmful strings) | Agent Smith (2402.08567) | Harmful string targets for jailbreak propagation |
| Schwartz Value Survey (56 values) | ValueFlow (2602.08567) | Human value categories for perturbation testing |

---

## Code Repositories (11 cloned)

All repositories are stored in `code/` directory with `--depth 1` shallow clones.

### Directly Related to Papers

| Repository | Stars | Paper | Description | Location |
|------------|-------|-------|-------------|----------|
| `sail-sg/Agent-Smith` | 118 | Agent Smith (2402.08567) | ICML 2024: Infectious jailbreak via adversarial images in MLLM agents | `code/Agent-Smith/` |
| `XHMY/AutoDefense` | 67 | AutoDefense (2403.04783) | Multi-agent LLM defense against jailbreak attacks | `code/AutoDefense/` |

### Highly Relevant Implementations

| Repository | Stars | Description | Location |
|------------|-------|-------------|----------|
| `microsoft/BIPIA` | - | Benchmark for Indirect Prompt Injection Attacks (Microsoft Research) | `code/BIPIA/` |
| `harshwt/Cross-Agent-Prompt-Injection` | - | Cross-LLM infection in multi-agent AI systems | `code/Cross-Agent-Prompt-Injection/` |
| `StavC/Here-Comes-the-AI-Worm` | - | AI worm propagation across AI agents | `code/Here-Comes-the-AI-Worm/` |
| `liu00222/Open-Prompt-Injection` | - | Open framework for prompt injection research | `code/Open-Prompt-Injection/` |
| `Greysahy/ipiguard` | 16 | EMNLP 2025 Oral: Tool dependency graph defense against indirect injection | `code/ipiguard/` |
| `aliasad059/RedDebate` | - | Multi-agent debate with adversarial prompts, short/long-term memory | `code/RedDebate/` |

### Supporting Implementations

| Repository | Stars | Description | Location |
|------------|-------|-------------|----------|
| `WamboDNS/PIGAN` | - | Adversarial prompt injection training via RL | `code/PIGAN/` |
| `pasquini-dario/project_mantis` | - | "Hacking Back the AI-Hacker" - defensive prompt injection | `code/project_mantis/` |
| `mariacwit/BSPS6` | 1 | Evaluating LLM-based multi-agent defense against jailbreak | `code/BSPS6-MultiAgent-Jailbreak-Defense/` |

---

## Paper Search Results

Raw search results from paper-finder are stored in `paper_search_results/`:
- `adversarial_prompt_injection_multi-agent_LLM_systems_20260227_231642.jsonl`
- `prompt_injection_propagation_attack_language_model_agents_20260227_232018.jsonl`
- `multi-agent_LLM_security_collaborative_reasoning_safety_20260227_232238.jsonl`

Total papers found: 645 across 3 search queries (103 with relevance >= 2, 27 core papers about injection + multi-agent systems).

---

## Key Findings from Deep Reading

### Propagation Models
1. **Logistic Growth** (Prompt Infection): Bounded infection spread in agent populations
2. **SIR Epidemiological** (Agent Smith): α infection symptom rate, β transmission rate, γ recovery rate; containment when β ≤ 2γ
3. **ACPM** (TOMA): Topology-dependent contamination spread with hierarchical payload encapsulation
4. **β-susceptibility** (ValueFlow): DAG-based perturbation cascade measurement

### Key Metrics
- Attack Success Rate (ASR): 40-78% depending on topology (TOMA)
- Amplification: 28.14% single-agent to 80.34% MAD (185% increase)
- Scale: 1 million agents infected in 27-31 rounds (Agent Smith)
- Defense: 94.8% blocking rate (T-Guard), ~50% reduction via LLM Tagging

### Models Tested Across Papers
- GPT-4o, GPT-4, GPT-3.5-turbo (most common)
- LLaVA-1.5 (multimodal agents)
- DeepSeek, Qwen3-8B, LLaMA-3.3-70B, Gemma-3-27B (open-source)

---

## Directory Structure

```
.
├── literature_review.md          # Comprehensive literature synthesis
├── resources.md                  # This file - resource catalog
├── arxiv_ids.json                # ArXiv IDs for all target papers
├── pyproject.toml                # Project configuration
├── papers/                       # Downloaded PDF papers (17 files)
│   ├── README.md                 # Paper index with descriptions
│   └── pages/                    # Chunked PDFs for deep reading
├── datasets/                     # Dataset information
│   ├── README.md                 # Dataset catalog with download instructions
│   └── dataset_search_results.json  # Full HuggingFace search results
├── code/                         # Cloned repositories (11 repos)
│   ├── README.md                 # Repository descriptions
│   ├── github_search_results.json   # Full GitHub search results
│   ├── Agent-Smith/              # ICML 2024 infectious jailbreak
│   ├── AutoDefense/              # Multi-agent LLM defense
│   ├── BIPIA/                    # Microsoft benchmark for indirect injection
│   ├── Cross-Agent-Prompt-Injection/  # Cross-LLM infection attacks
│   ├── Here-Comes-the-AI-Worm/  # AI worm propagation
│   ├── Open-Prompt-Injection/    # Open prompt injection framework
│   ├── ipiguard/                 # EMNLP 2025 tool-graph defense
│   ├── RedDebate/                # Adversarial multi-agent debate
│   ├── PIGAN/                    # Adversarial injection training via RL
│   ├── project_mantis/           # Defensive prompt injection
│   └── BSPS6-MultiAgent-Jailbreak-Defense/  # Multi-agent jailbreak defense eval
└── paper_search_results/         # Raw paper-finder search results (JSONL)
```
