# Literature Review: Adversarial Prompt Propagation Dynamics in Multi-Agent LLM Collaborative Systems

## Research Hypothesis
Adversarial prompt injections in multi-agent LLM systems exhibit emergent propagation patterns that amplify through collaborative reasoning chains, with injection persistence and mutation rates correlating positively with task complexity and agent interaction frequency.

---

## 1. Introduction and Scope

This literature review synthesizes research on adversarial prompt injection attacks in multi-agent large language model (LLM) systems, with particular focus on propagation dynamics, topological effects, and defense mechanisms. The review covers 17 papers spanning 2023-2026, organized into four thematic areas: (1) attack propagation models, (2) multi-agent debate vulnerabilities, (3) topology-aware attacks, and (4) defense frameworks.

---

## 2. Attack Propagation Models

### 2.1 Self-Replicating Prompt Injection (Prompt Infection)

**Lee & Tiwari (2024)** [[2410.07283](https://arxiv.org/abs/2410.07283)] introduce the concept of self-replicating prompt injection in multi-agent systems, directly analogous to biological viral infection. Their attack framework consists of four components:

- **Prompt Hijacking**: Overrides the agent's original system prompt
- **Payload**: The malicious instruction to execute
- **Data Stealing**: Exfiltrates information from compromised agents
- **Self-Replication**: The infected agent propagates the injection to other agents it communicates with

Key findings:
- Infection spread follows a **logistic growth pattern** in social simulations with multiple agents
- Stronger models (GPT-4o) are paradoxically *more dangerous* when compromised, as they are better at executing the self-replication instructions while appearing normal
- A dataset of 120 user instructions x 3 tool types = 360 instruction-tool pairs was used for evaluation
- The **LLM Tagging** defense (prepending markers to distinguish user vs. agent-generated content) reduces attack success from ~80% to ~40%

### 2.2 Infectious Jailbreak via Adversarial Images (Agent Smith)

**Gu et al. (2024)** [[2402.08567](https://arxiv.org/abs/2402.08567)] demonstrate that a single adversarial image can trigger exponential propagation of jailbreak across multimodal LLM (MLLM) agent populations. Their key contributions:

- **Epidemiological modeling**: The infection dynamics follow an **SIR-like model** with parameters:
  - α (infection symptom rate): probability a compromised agent exhibits harmful behavior
  - β (transmission rate): probability of successful infection per contact
  - γ (recovery rate): probability of an agent recovering from infection
- **Scale**: A single image can infect **1 million agents in 27-31 rounds** of communication
- **Defense criterion**: Infection is contained when β ≤ 2γ (transmission rate must be at most twice the recovery rate)
- Tested on LLaVA-1.5 agents using the AdvBench dataset (574 harmful strings)
- The adversarial image is optimized to maximize jailbreak transferability across diverse agent interactions

### 2.3 Value Perturbation Propagation (ValueFlow)

**ValueFlow (2026)** [[2602.08567](https://arxiv.org/abs/2602.08567)] provides a formal framework for measuring how value perturbations propagate through multi-agent systems:

- **β-susceptibility metric**: Measures how susceptible individual agents are to value drift when exposed to perturbed inputs
- **System Susceptibility (SS)**: Aggregate metric capturing the system-level vulnerability
- Uses a **DAG (Directed Acyclic Graph) model** for multi-agent interaction topology
- Evaluates across 5 models: Qwen3-8B, LLaMA-3.3-70B, GPT-3.5-Turbo, GPT-4o, Gemma-3-27B
- Dataset derived from **Schwartz Value Survey** (56 human values)
- Demonstrates that value perturbations amplify through multi-hop agent chains

### 2.4 Data Leakage Through Orchestrator Networks (OMNI-LEAK)

**OMNI-LEAK (2026)** [[2602.13477](https://arxiv.org/abs/2602.13477)] explores data leakage attacks in orchestrator-based multi-agent networks, showing how adversarial prompts can be crafted to exfiltrate sensitive information across agent boundaries through the orchestration layer.

---

## 3. Multi-Agent Debate Vulnerabilities

### 3.1 Amplified Jailbreak in Multi-Agent Debate

**Amplified Vulnerabilities (2025)** [[2504.16489](https://arxiv.org/abs/2504.16489)] demonstrates that multi-agent debate (MAD) systems are significantly more vulnerable to structured jailbreak attacks than single-agent systems. Their attack combines four techniques:

1. **Narrative Encapsulation**: Wrapping malicious prompts in fictional scenarios
2. **Role-Driven Escalation**: Assigning agents specific roles that normalize harmful content
3. **Iterative Refinement**: Progressively escalating content across debate rounds
4. **Rhetorical Obfuscation**: Using persuasive language to bypass safety filters

Key results:
- Attack success rate increases from **28.14% (single-agent) to 80.34% (MAD)** - a 185% amplification
- The debate mechanism itself amplifies harmful content through iterative refinement
- Tested on GPT-4o, GPT-4, GPT-3.5-turbo, and DeepSeek models
- Demonstrates that collaborative reasoning chains act as an amplification mechanism for adversarial content

### 3.2 Conformity-Driven Injection (MAD-Spear)

**MAD-Spear (2025)** [[2507.13038](https://arxiv.org/abs/2507.13038)] introduces a conformity-driven prompt injection attack exploiting LLMs' tendency toward herd behavior in debate systems:

- Inspired by **Sybil attacks** in distributed systems - a single compromised agent can influence the group
- Provides a formal definition of **MAD fault-tolerance**: the minimum number of compromised agents needed to consistently influence debate outcomes
- Even **1 out of 6 compromised agents** has strong impact on debate outcomes due to conformity pressure
- LLMs exhibit systematic bias toward agreeing with majority positions, which adversarial agents exploit
- The attack is particularly effective because it leverages the intended collaborative mechanism (consensus-seeking) as the attack vector

---

## 4. Topology-Aware Attacks and Defenses

### 4.1 Multi-Hop Topology Exploitation (TOMA)

**Tipping the Dominos (2025)** [[2512.04129](https://arxiv.org/abs/2512.04129)] introduces TOMA (Topology-Aware Multi-Hop Attack), the first attack framework that explicitly exploits multi-agent system topology:

- **Adversarial Contamination Propagation Model (ACPM)**: Formal model of how contamination spreads through agent networks based on topological structure
- **Hierarchical Payload Encapsulation Scheme (HPES)**: Recursive payload wrapping for multi-hop propagation - each layer is designed for the specific agent it passes through
- Tested across **5 topologies** on 3 frameworks (MAGENTIC-ONE, LANGMANUS, OWL):
  - Chain, Star, Tree, Mesh, Hierarchical
  - ASR ranges from **40-78%** depending on topology
- **T-Guard defense**: A topology-trust framework that achieves **94.8% blocking rate** by:
  - Assigning trust scores based on agent position in topology
  - Monitoring message content at topological choke points
  - Isolating potentially compromised communication paths

### 4.2 Topological Safety Analysis (NetSafe)

**NetSafe (2024)** [[2410.15686](https://arxiv.org/abs/2410.15686)] provides a systematic exploration of how network topology affects safety in multi-agent LLM systems:

- Analyzes safety properties across different network structures (scale-free, small-world, random, etc.)
- Demonstrates that certain topological features (high centrality nodes, short path lengths) increase vulnerability to attack propagation
- Provides quantitative metrics for assessing topological safety

### 4.3 Communication Topology Design (G-Designer)

**G-Designer (2024)** [[2410.11782](https://arxiv.org/abs/2410.11782)] uses graph neural networks to architect optimal communication topologies for multi-agent systems, with implications for both performance and security:

- Demonstrates that topology design significantly impacts system robustness
- Provides a framework for balancing task performance with communication efficiency
- Relevant to defense design: optimal topologies can limit attack propagation paths

---

## 5. Defense Frameworks

### 5.1 Multi-Agent Defense (AutoDefense)

**AutoDefense (2024)** [[2403.04783](https://arxiv.org/abs/2403.04783)] proposes using multi-agent architectures themselves as a defense against jailbreak attacks:

- Employs a multi-agent pipeline to filter and validate LLM responses before delivery
- Agents check each other's outputs for signs of jailbreak or harmful content
- Demonstrates that collaborative defense can be more robust than single-agent safety measures

### 5.2 Unknown Attack Defense (BlindGuard)

**BlindGuard (2025)** [[2508.08127](https://arxiv.org/abs/2508.08127)] addresses defense against unknown attacks in multi-agent systems:

- Designed to work without prior knowledge of the specific attack type
- Uses anomaly detection principles to identify compromised agent behavior
- Provides robustness guarantees against novel attack vectors

### 5.3 Co-Evolutionary Safety (Evo-MARL)

**Evo-MARL (2025)** [[2508.03864](https://arxiv.org/abs/2508.03864)] introduces a co-evolutionary multi-agent reinforcement learning approach to internalized safety:

- Agents learn safety behaviors through co-evolutionary dynamics
- Attack and defense strategies evolve together, leading to more robust safety
- Demonstrates that adaptive defense mechanisms outperform static safety rules

### 5.4 Predictive Security Monitoring (AgentMonitor)

**AgentMonitor (2024)** [[2408.14972](https://arxiv.org/abs/2408.14972)] provides a plug-and-play framework for monitoring multi-agent systems:

- Predictive security that anticipates potential attacks
- Runtime monitoring of agent interactions for anomalous patterns
- Can be integrated into existing multi-agent frameworks without major modifications

### 5.5 Prompt Injection Detection

**Liu et al. (2023)** [[2310.12815](https://arxiv.org/abs/2310.12815)] and related work on prompt injection detection provide foundational approaches to identifying malicious prompts:

- Classifier-based detection of prompt injection attempts
- Analysis of prompt injection attack patterns and taxonomies
- Baseline defense mechanisms that can be extended to multi-agent settings

---

## 6. Foundational Work

### 6.1 Indirect Prompt Injection

**Greshake et al. (2023)** [[2302.12173](https://arxiv.org/abs/2302.12173)] established the foundational framework for indirect prompt injection, demonstrating how LLM-integrated applications can be compromised through injected content in external data sources. This work is the conceptual predecessor to multi-agent prompt propagation attacks.

### 6.2 Covert Injection in MLLM Societies (Wolf Within)

**The Wolf Within (2024)** [[2402.14859](https://arxiv.org/abs/2402.14859)] examines covert injection of malice into MLLM societies through a single compromised agent operating as a covert operative, demonstrating the outsized impact a single adversarial node can have on the overall system behavior.

### 6.3 Multi-Agent Resilience

**Resilience of Multi-Agent LLM Collaboration (2024)** [[2408.00989](https://arxiv.org/abs/2408.00989)] studies how multi-agent systems respond to faulty or adversarial agents, providing baseline resilience metrics and demonstrating failure modes in collaborative settings.

---

## 7. Synthesis and Key Themes

### 7.1 Propagation Models Converge on Epidemiological Frameworks

Multiple papers independently adopt epidemiological models to describe adversarial prompt propagation:
- **Logistic growth** (Prompt Infection) for bounded populations
- **SIR models** (Agent Smith) for infection-recovery dynamics
- **ACPM** (TOMA) for topology-dependent contamination spread
- **β-susceptibility** (ValueFlow) for perturbation cascades

This convergence suggests that epidemiological modeling is a natural and productive framework for understanding adversarial propagation in multi-agent systems.

### 7.2 Collaborative Mechanisms as Attack Amplifiers

A critical finding across multiple papers is that the very mechanisms designed for collaborative reasoning amplify adversarial content:
- Debate mechanisms amplify jailbreak success by 185% (Amplified Vulnerabilities)
- Consensus-seeking behavior creates conformity pressure exploitable by single compromised agents (MAD-Spear)
- Self-replication leverages agents' instruction-following capability (Prompt Infection)
- Multi-hop communication enables recursive payload encapsulation (TOMA)

### 7.3 Topology as a Critical Factor

Network topology emerges as a key determinant of both vulnerability and defense effectiveness:
- Attack success varies significantly across topologies (40-78% ASR, TOMA)
- Centrality metrics predict vulnerability (NetSafe)
- Topology-aware defenses achieve high blocking rates (94.8%, T-Guard)
- Communication topology design can be optimized for robustness (G-Designer)

### 7.4 Stronger Models, Greater Risk

The "capability paradox" appears across multiple studies:
- GPT-4o is more dangerous when compromised than GPT-3.5-turbo (Prompt Infection)
- More capable models are better at executing self-replication and social engineering
- This suggests that advancing model capabilities may increase multi-agent system vulnerability unless safety measures scale accordingly

### 7.5 Defense Landscape

Current defenses can be categorized into:
1. **Detection-based**: Classify injected content (prompt injection classifiers, AgentMonitor)
2. **Structural**: Modify communication topology to limit propagation (T-Guard, G-Designer)
3. **Collaborative**: Use multi-agent verification to catch attacks (AutoDefense)
4. **Tagging-based**: Mark message provenance to distinguish user/agent content (LLM Tagging)
5. **Evolutionary**: Co-evolve attack and defense strategies (Evo-MARL)
6. **Anomaly-based**: Detect unknown attacks through behavioral analysis (BlindGuard)

---

## 8. Research Gaps and Opportunities

Based on this review, several gaps directly relevant to our hypothesis emerge:

1. **Mutation dynamics**: No existing work systematically studies how adversarial prompts mutate as they propagate through agent chains. The hypothesis's claim about mutation rates is largely untested.

2. **Task complexity correlation**: While papers show that collaborative mechanisms amplify attacks, the specific correlation between task complexity and injection persistence has not been quantified.

3. **Interaction frequency effects**: The relationship between agent interaction frequency and propagation dynamics remains unexplored beyond simple epidemic models.

4. **Cross-framework generalization**: Most studies test on a single multi-agent framework; comparative analysis across frameworks (AutoGen, LangGraph, CrewAI, etc.) is lacking.

5. **Long-horizon propagation**: Existing studies focus on relatively short interaction sequences; the dynamics of adversarial propagation over extended collaborative sessions remain unknown.

6. **Emergent behaviors**: The hypothesis predicts "emergent propagation patterns" - this requires experimental designs that can detect and characterize emergent phenomena in adversarial multi-agent interactions.

---

## 9. Methodological Patterns

### Common Evaluation Metrics
- **Attack Success Rate (ASR)**: Standard metric across all attack papers
- **Infection rate over rounds**: Used by epidemiological models
- **Blocking/detection rate**: Used by defense papers
- **β-susceptibility / System Susceptibility**: ValueFlow-specific metrics

### Common Models Tested
- GPT-4o, GPT-4, GPT-3.5-turbo (most common)
- LLaVA-1.5 (multimodal)
- DeepSeek, Qwen3-8B, LLaMA-3.3-70B, Gemma-3-27B (open-source)

### Common Experimental Setups
- Social simulation with multiple communicating agents
- Multi-agent debate with defined round structures
- Tool-using agents with structured task pipelines
- Orchestrator-based hierarchical agent networks

---

## 10. References

1. Lee, S. & Tiwari, A. (2024). Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems. arXiv:2410.07283
2. Gu, X. et al. (2024). Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents. arXiv:2402.08567
3. (2025). Tipping the Dominos: Topology-Aware Multi-Hop Attacks on LLM-Based Multi-Agent Systems. arXiv:2512.04129
4. (2026). ValueFlow: Measuring the Propagation of Value Perturbations in Multi-Agent LLM Systems. arXiv:2602.08567
5. (2025). Amplified Vulnerabilities: Structured Jailbreak Attacks on LLM-based Multi-Agent Debate. arXiv:2504.16489
6. (2025). MAD-Spear: A Conformity-Driven Prompt Injection Attack on Multi-Agent Debate Systems. arXiv:2507.13038
7. (2026). OMNI-LEAK: Orchestrator Multi-Agent Network Induced Data Leakage. arXiv:2602.13477
8. (2024). AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks. arXiv:2403.04783
9. (2024). NetSafe: Exploring the Topological Safety of Multi-agent Networks. arXiv:2410.15686
10. (2025). BlindGuard: Safeguarding LLM-based Multi-Agent Systems under Unknown Attacks. arXiv:2508.08127
11. (2025). Evo-MARL: Co-Evolutionary Multi-Agent Reinforcement Learning for Internalized Safety. arXiv:2508.03864
12. (2024). AgentMonitor: A Plug-and-Play Framework for Predictive and Secure Multi-Agent Systems. arXiv:2408.14972
13. (2024). G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks. arXiv:2410.11782
14. Greshake, K. et al. (2023). Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. arXiv:2302.12173
15. Liu, Y. et al. (2023). Prompt Injection Attacks and Defenses in LLM-Integrated Applications. arXiv:2310.12815
16. (2024). The Wolf Within: Covert Injection of Malice into MLLM Societies. arXiv:2402.14859
17. (2024). On the Resilience of Multi-Agent LLM Collaboration with Faulty Agents. arXiv:2408.00989
