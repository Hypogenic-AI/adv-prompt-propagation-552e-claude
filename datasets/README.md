# Datasets

This directory contains dataset search results and download instructions for datasets relevant to **Adversarial Prompt Propagation Dynamics in Multi-Agent LLM Collaborative Systems**.

## Search Results

`dataset_search_results.json` contains 75 unique datasets found across 9 search queries on the HuggingFace Datasets API.

## Recommended Datasets

### HIGH Priority

**1. JailbreakBench/JBB-Behaviors** (16,972 downloads)
- NeurIPS 2024 Datasets and Benchmarks Track
- Standardized harmful behaviors for jailbreak evaluation
- License: MIT
- Download: `pip install datasets && python -c "from datasets import load_dataset; ds = load_dataset('JailbreakBench/JBB-Behaviors')"`

**2. walledai/AdvBench** (10,170 downloads)
- AdvBench harmful behaviors/strings used by GCG attack and Agent Smith papers
- License: MIT
- Download: `python -c "from datasets import load_dataset; ds = load_dataset('walledai/AdvBench')"`

**3. deepset/prompt-injections** (6,183 downloads)
- Binary-labeled prompt injection classification (546 examples)
- Fields: `text` (string), `label` (0=safe, 1=injection)
- License: Apache-2.0
- Download: `python -c "from datasets import load_dataset; ds = load_dataset('deepset/prompt-injections')"`

**4. xTRam1/safe-guard-prompt-injection** (1,573 downloads)
- Large prompt injection dataset (10K-100K examples)
- Download: `python -c "from datasets import load_dataset; ds = load_dataset('xTRam1/safe-guard-prompt-injection')"`

### MEDIUM Priority

**5. TrustAIRLab/in-the-wild-jailbreak-prompts** (1,826 downloads)
- Real-world jailbreak prompts collected in the wild
- Download: `python -c "from datasets import load_dataset; ds = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts')"`

**6. jackhhao/jailbreak-classification** (1,629 downloads)
- Labeled jailbreak classification dataset
- Download: `python -c "from datasets import load_dataset; ds = load_dataset('jackhhao/jailbreak-classification')"`

**7. Lakera/mosscap_prompt_injection** (1,208 downloads)
- Large-scale (100K-1M) prompt injection dataset from Lakera AI
- Download: `python -c "from datasets import load_dataset; ds = load_dataset('Lakera/mosscap_prompt_injection')"`

**8. walledai/JailbreakHub** (254 downloads)
- 15,140+ jailbreak prompts from Reddit, Discord, websites
- From paper: "Do Anything Now" (arXiv:2308.03825)
- License: MIT
- Download: `python -c "from datasets import load_dataset; ds = load_dataset('walledai/JailbreakHub')"`

**9. ivnle/advbench_harmful_strings** (4 downloads)
- Exact harmful strings subset used in Agent Smith attack paper
- Download: `python -c "from datasets import load_dataset; ds = load_dataset('ivnle/advbench_harmful_strings')"`

### LOW Priority

**10. facebook/cyberseceval3-visual-prompt-injection** (540 downloads)
- Meta CyberSecEval3 visual prompt injection for multi-modal attack vectors
- Download: `python -c "from datasets import load_dataset; ds = load_dataset('facebook/cyberseceval3-visual-prompt-injection')"`

## Paper-Specific Datasets

These datasets are described within the papers but may need to be constructed from paper descriptions or obtained from paper authors:

| Dataset | Paper | Size | Description |
|---------|-------|------|-------------|
| Instruction-Tool pairs | Prompt Infection (2410.07283) | 360 pairs | 120 user instructions x 3 tool types |
| AdvBench harmful strings | Agent Smith (2402.08567) | 574 strings | Available as `walledai/AdvBench` |
| Schwartz Value Survey | ValueFlow (2602.08567) | 56 values | Human value categories for perturbation testing |

## Batch Download Script

To download all high-priority datasets:

```python
from datasets import load_dataset

datasets_to_download = [
    "JailbreakBench/JBB-Behaviors",
    "walledai/AdvBench",
    "deepset/prompt-injections",
    "xTRam1/safe-guard-prompt-injection",
]

for ds_name in datasets_to_download:
    print(f"Downloading {ds_name}...")
    ds = load_dataset(ds_name)
    print(f"  Loaded: {ds}")
```

## Notes

- Datasets are NOT downloaded to this directory to avoid large file storage in the repository
- Use the HuggingFace `datasets` library to download datasets on demand: `pip install datasets`
- The `dataset_search_results.json` file contains full metadata for all 75 discovered datasets
