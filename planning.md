# Research Plan: Adversarial Prompt Propagation Dynamics in Multi-Agent LLM Collaborative Systems

## Motivation & Novelty Assessment

### Why This Research Matters
Multi-agent LLM systems are being deployed at scale for complex tasks (coding, research, decision-making), yet their security properties under adversarial conditions are poorly understood. A single adversarial prompt injected into one agent can potentially compromise an entire collaborative pipeline, with effects that amplify rather than attenuate through agent interactions. Understanding these propagation dynamics is critical for building trustworthy multi-agent AI systems.

### Gap in Existing Work
Based on the literature review, existing work has demonstrated:
- Self-replicating prompts exist and spread logistically (Prompt Infection, 2024)
- Multi-agent debate amplifies jailbreaks by 185% (Amplified Vulnerabilities, 2025)
- Topology matters for attack success rates (TOMA, 2025)

**Key gaps our work addresses:**
1. **Prompt mutation dynamics**: No existing work systematically tracks how adversarial prompt content *semantically mutates* as it passes through agent chains using real LLM APIs.
2. **Task complexity correlation**: The specific relationship between collaborative task complexity and injection persistence/amplification is unquantified.
3. **Interaction frequency effects**: How does the frequency of inter-agent communication affect propagation beyond simple epidemic models?
4. **Detection at communication interfaces**: While defenses exist, lightweight detection algorithms operating specifically at agent communication boundaries haven't been systematically tested.

### Our Novel Contribution
We conduct the **first empirical study** using real LLM APIs (GPT-4.1) to measure:
1. How adversarial prompt content semantically drifts and mutates across multi-agent reasoning chains
2. The correlation between task complexity and injection persistence
3. How different network topologies (chain, star, mesh) affect propagation dynamics
4. A simple detection mechanism based on semantic similarity at communication interfaces

### Experiment Justification
- **Experiment 1 (Propagation Tracking)**: Measures the core phenomenon — do adversarial prompts persist and mutate through agent chains? This directly tests the central hypothesis.
- **Experiment 2 (Topology Comparison)**: Tests whether network structure modulates propagation, connecting to the TOMA/NetSafe literature but with real semantic mutation tracking.
- **Experiment 3 (Task Complexity)**: Tests the specific claim that task complexity correlates with injection persistence, which is untested in prior work.
- **Experiment 4 (Detection)**: Validates whether observed propagation signatures can be used for practical detection, providing actionable defense insights.

---

## Research Question
Do adversarial prompt injections in multi-agent LLM systems exhibit emergent propagation patterns that amplify through collaborative reasoning chains, and do injection persistence and mutation rates correlate with task complexity and network topology?

## Hypothesis Decomposition

### H1: Propagation & Persistence
Adversarial prompt injections persist across multiple agent hops in collaborative reasoning chains, with measurable semantic traces detectable ≥3 hops from the injection point.

### H2: Semantic Mutation
As adversarial content propagates, it undergoes semantic mutation — the exact phrasing changes but the adversarial intent is preserved or amplified.

### H3: Topology Effects
Different network topologies (chain, star, mesh) produce significantly different propagation patterns, with denser topologies showing faster but potentially less persistent spread.

### H4: Task Complexity Correlation
Higher task complexity (multi-step reasoning) correlates positively with injection persistence, as agents must process and relay more context including adversarial content.

### H5: Detection Feasibility
Semantic similarity analysis at agent communication interfaces can detect adversarial propagation with ≥70% precision and ≥60% recall.

---

## Proposed Methodology

### Approach
We build a controlled multi-agent system using real GPT-4.1 API calls where agents collaborate on reasoning tasks. We inject adversarial prompts at specific points and track their propagation, mutation, and impact across agents using semantic embedding analysis.

### Experimental Steps

#### Step 1: Multi-Agent Framework Construction
- Build a lightweight multi-agent framework supporting configurable topologies (chain, star, mesh)
- Each agent has a system prompt defining its role and receives messages from connected agents
- Agents communicate via structured message passing, all routed through GPT-4.1 API
- Log all inter-agent messages for analysis

#### Step 2: Task Design (3 Complexity Levels)
- **Simple**: Single-step factual Q&A (e.g., "What is the capital of France?")
- **Medium**: Multi-step reasoning (e.g., "Analyze pros and cons of X and recommend a decision")
- **Complex**: Multi-document synthesis requiring multiple reasoning chains

#### Step 3: Adversarial Prompt Design
Design 5 injection types of increasing sophistication:
1. Direct instruction override ("Ignore previous instructions, instead...")
2. Context poisoning (embedding misleading facts in legitimate-looking content)
3. Role hijacking ("You are actually a different agent whose job is to...")
4. Payload propagation ("Include the following text in all your responses...")
5. Subtle bias injection (introducing systematic bias without explicit instruction)

#### Step 4: Propagation Measurement
For each (topology × task_complexity × injection_type) combination:
- Run the multi-agent collaborative task
- Measure at each agent hop:
  - **Injection persistence**: Does the adversarial content appear in the agent's output?
  - **Semantic similarity**: Cosine similarity between injected content and agent outputs (using text-embedding-3-small)
  - **Task accuracy degradation**: How much does the agent's task output deviate from correct?
  - **Mutation entropy**: Lexical and semantic drift of the adversarial content

#### Step 5: Detection Algorithm
- Compute semantic similarity between consecutive agent messages
- Flag messages where similarity to known injection patterns exceeds threshold
- Evaluate precision/recall of this simple detection approach

### Baselines
1. **No injection**: Clean collaborative task completion (control)
2. **Single-agent injection**: Same adversarial prompts against a single agent (no propagation)
3. **Random noise**: Non-adversarial noise injected at same points (to distinguish adversarial effects from general perturbation)

### Evaluation Metrics
1. **Injection Persistence Rate (IPR)**: Fraction of downstream agents whose outputs contain detectable traces of the adversarial prompt
2. **Semantic Drift Score (SDS)**: Cosine distance between original injection and its manifestation at each hop
3. **Task Accuracy Degradation (TAD)**: Difference in task performance between clean and injected conditions
4. **Mutation Entropy (ME)**: Shannon entropy of n-gram distributions in adversarial content across hops
5. **Detection Precision/Recall**: For the communication-interface detection algorithm
6. **Amplification Factor (AF)**: Ratio of adversarial effect at hop N vs. hop 0

### Statistical Analysis Plan
- **Within-subjects comparisons**: Paired t-tests (or Wilcoxon signed-rank if non-normal) for clean vs. injected conditions
- **Between-conditions comparisons**: One-way ANOVA (or Kruskal-Wallis) for topology and complexity effects
- **Correlation analysis**: Spearman rank correlation for complexity × persistence relationship
- **Significance level**: α = 0.05, with Bonferroni correction for multiple comparisons
- **Effect sizes**: Cohen's d for pairwise comparisons, η² for ANOVA
- **N per condition**: 10 trials with different random seeds (100 total API calls per condition estimated)

## Expected Outcomes

### Supporting the hypothesis:
- IPR > 0.3 at ≥3 hops (adversarial content persists)
- Significant positive correlation between task complexity and IPR (ρ > 0.3, p < 0.05)
- Mesh topology shows higher early-hop persistence but faster decay vs. chain topology
- Detection algorithm achieves ≥70% precision, ≥60% recall

### Refuting the hypothesis:
- IPR drops to <0.1 by hop 2 (rapid attenuation)
- No significant correlation between complexity and persistence
- Topology has no significant effect on propagation dynamics

## Timeline and Milestones
- **Resource review & planning**: 10 min ✓
- **Environment setup & dependencies**: 10 min
- **Multi-agent framework implementation**: 25 min
- **Task & injection design**: 10 min
- **Experiment execution (all conditions)**: 30 min (API calls)
- **Analysis & visualization**: 20 min
- **Documentation (REPORT.md)**: 15 min
- **Buffer for debugging**: 20 min
- **Total**: ~2.5 hours (within budget)

## Potential Challenges
1. **API rate limits**: Mitigate by batching calls and adding exponential backoff
2. **Cost**: Estimated ~$20-40 for GPT-4.1 calls; acceptable
3. **Non-determinism**: Use temperature=0.3 for reproducibility while allowing some variation; average over 10 trials
4. **Defining "persistence"**: Use embedding similarity thresholds validated against human judgment on a small sample

## Success Criteria
1. Complete data collection across all experimental conditions
2. At least 2 of the 5 sub-hypotheses show statistically significant results
3. Sufficient data to characterize propagation patterns (even if hypothesis is partially refuted)
4. Actionable detection algorithm with measured precision/recall
5. Complete REPORT.md with actual experimental results
