# Code Repositories

This directory contains 11 cloned repositories relevant to **Adversarial Prompt Propagation Dynamics in Multi-Agent LLM Collaborative Systems**.

All repositories were cloned with `--depth 1` (shallow clone) to minimize disk usage.

## Repository Index

### Paper Implementations (Directly from Papers)

**1. Agent-Smith** (`sail-sg/Agent-Smith`) - 118 stars
- Paper: Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents (ICML 2024)
- ArXiv: [2402.08567](https://arxiv.org/abs/2402.08567)
- Contents: Adversarial image optimization, LLaVA-based agent simulation, SIR infection dynamics
- Language: Python
- URL: https://github.com/sail-sg/Agent-Smith

**2. AutoDefense** (`XHMY/AutoDefense`) - 67 stars
- Paper: AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks
- ArXiv: [2403.04783](https://arxiv.org/abs/2403.04783)
- Contents: Multi-agent defense pipeline, jailbreak filtering, collaborative verification
- Language: Python
- URL: https://github.com/XHMY/AutoDefense

### Prompt Injection Attack Frameworks

**3. BIPIA** (`microsoft/BIPIA`)
- Paper: Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models
- ArXiv: [2312.14197](https://arxiv.org/abs/2312.14197)
- Contents: Microsoft Research benchmark for indirect prompt injection; attack and defense implementations
- Language: Python
- URL: https://github.com/microsoft/BIPIA

**4. Cross-Agent-Prompt-Injection** (`harshwt/Cross-Agent-Prompt-Injection-Attacks-CA-PIA-`)
- Description: Cross-LLM infection in multi-agent AI systems where malicious instructions propagate between agents
- Contents: Attack vectors for cross-agent infection, propagation analysis
- Language: Python
- URL: https://github.com/harshwt/Cross-Agent-Prompt-Injection-Attacks-CA-PIA-

**5. Here-Comes-the-AI-Worm** (`StavC/Here-Comes-the-AI-Worm`)
- Description: AI worm that propagates across AI agents through adversarial self-replicating prompts
- Contents: Worm implementation, propagation experiments, GenAI ecosystem attack simulation
- Language: Python
- URL: https://github.com/StavC/Here-Comes-the-AI-Worm

**6. Open-Prompt-Injection** (`liu00222/Open-Prompt-Injection`)
- Description: Open framework for prompt injection attack research and evaluation
- Contents: Attack implementations, defense baselines, evaluation framework
- Language: Python
- URL: https://github.com/liu00222/Open-Prompt-Injection

### Defense Implementations

**7. ipiguard** (`Greysahy/ipiguard`) - 16 stars
- Paper: IPIGuard: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection (EMNLP 2025 Oral)
- Contents: Tool dependency graph construction, injection detection via graph analysis
- Language: Python
- URL: https://github.com/Greysahy/ipiguard

**8. project_mantis** (`pasquini-dario/project_mantis`)
- Description: "Hacking Back the AI-Hacker" - Defensive prompt injection techniques
- Contents: Counter-attack strategies against prompt injection, defensive payloads
- Language: Python
- URL: https://github.com/pasquini-dario/project_mantis

**9. BSPS6-MultiAgent-Jailbreak-Defense** (`mariacwit/BSPS6`) - 1 star
- Description: Evaluating LLM-based multi-agent defense against jailbreak attacks
- Contents: Multi-agent defense evaluation framework
- Language: Python
- URL: https://github.com/mariacwit/BSPS6

### Adversarial Training & Red Teaming

**10. PIGAN** (`WamboDNS/PIGAN`)
- Description: PiGAN - Adversarial Prompt Injection Training via reinforcement learning
- Contents: RL-based adversarial training for discovering prompt injection vulnerabilities
- Language: Python
- URL: https://github.com/WamboDNS/PIGAN

**11. RedDebate** (`aliasad059/RedDebate`)
- Description: Multi-agent debate with adversarial prompts, using short-term and long-term memory
- Contents: Debate orchestration, adversarial prompt generation, memory-augmented agents
- Language: Python
- URL: https://github.com/aliasad059/RedDebate

## GitHub Search Results

Full search results from the GitHub API are stored in `github_search_results.json`, containing metadata for 40+ repositories found across 6 search queries.

## Usage Notes

- All repos use shallow clones (`--depth 1`). To get full history: `git fetch --unshallow`
- Most repos require Python 3.8+ and common ML libraries (transformers, torch, etc.)
- Check each repo's README.md for specific setup instructions and dependencies
- Some repos may require API keys for LLM services (OpenAI, etc.)
