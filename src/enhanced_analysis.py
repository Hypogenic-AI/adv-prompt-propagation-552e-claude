"""
Enhanced analysis focusing on deeper insights from the experimental data.

Complements the main analysis with:
1. Per-agent hop-level propagation analysis
2. Marker-based detection (the effective approach)
3. Qualitative examples of prompt mutation
4. Enhanced detection algorithm using marker + semantic features
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results" / "data"
PLOTS_DIR = Path(__file__).parent.parent / "results" / "plots"


def load_results():
    with open(RESULTS_DIR / "experiment_results.json") as f:
        return json.load(f)


def analyze_per_agent_propagation(results):
    """Analyze propagation at the individual agent level within each topology."""
    print("\n" + "=" * 60)
    print("PER-AGENT PROPAGATION ANALYSIS")
    print("=" * 60)

    # For chain topology, look at how injections propagate hop by hop
    chain_results = [r for r in results
                     if r["topology"] == "chain" and r["injection_type"] != "clean"]

    agent_order = ["Agent_0_Researcher", "Agent_1_Critic",
                   "Agent_2_Synthesizer", "Agent_3_Planner"]

    propagation_data = defaultdict(lambda: defaultdict(list))

    for r in chain_results:
        inj_type = r["injection_type"]
        persistence = r.get("persistence_metrics", {})
        for agent_name, info in persistence.items():
            # Determine agent position
            for idx, name in enumerate(agent_order):
                if name == agent_name:
                    propagation_data[inj_type][idx].append(info["semantic_similarity"])
                    break

    fig, ax = plt.subplots(figsize=(10, 6))
    for inj_type, pos_data in propagation_data.items():
        positions = sorted(pos_data.keys())
        means = [np.mean(pos_data[p]) for p in positions]
        stds = [np.std(pos_data[p]) for p in positions]
        ax.errorbar(positions, means, yerr=stds, marker='o', linewidth=2,
                   capsize=4, label=inj_type.replace("_", " "))

    ax.set_xticks(range(4))
    ax.set_xticklabels(["Researcher\n(injected)", "Critic", "Synthesizer", "Planner"])
    ax.set_title("Chain Topology: Semantic Similarity to Injection\nAcross Agent Positions")
    ax.set_xlabel("Agent Position in Chain")
    ax.set_ylabel("Semantic Similarity to Injection")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "chain_propagation_by_agent.png", dpi=150)
    plt.close()
    print("Saved: chain_propagation_by_agent.png")


def analyze_marker_persistence_detailed(results):
    """Detailed analysis of marker-based injection persistence."""
    print("\n" + "=" * 60)
    print("DETAILED MARKER PERSISTENCE ANALYSIS")
    print("=" * 60)

    # Build detailed marker data
    rows = []
    for r in results:
        if r["injection_type"] == "clean":
            continue
        inj_type = r["injection_type"]
        persistence = r.get("persistence_metrics", {})

        for agent_name, info in persistence.items():
            marker_hit = info["marker_detected"].get(inj_type, False)
            rows.append({
                "injection_type": inj_type,
                "topology": r["topology"],
                "complexity": r["complexity"],
                "agent": agent_name,
                "marker_detected": marker_hit,
                "semantic_sim": info["semantic_similarity"],
                "lexical_overlap": info["lexical_overlap"],
            })

    df = pd.DataFrame(rows)

    # Marker detection rate by injection type and topology
    pivot = df.groupby(["injection_type", "topology"])["marker_detected"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    ax = axes[0]
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                vmin=0, vmax=1)
    ax.set_title("Marker Detection Rate\n(Injection Type × Topology)")
    ax.set_xlabel("Topology")
    ax.set_ylabel("Injection Type")

    # By complexity
    ax = axes[1]
    complexity_order = ["simple", "medium", "complex"]
    for inj_type in ["direct_override", "context_poisoning", "payload_propagation", "subtle_bias"]:
        subset = df[df["injection_type"] == inj_type]
        means = subset.groupby("complexity")["marker_detected"].mean()
        means = means.reindex(complexity_order)
        ax.plot(complexity_order, means.values, marker='o', linewidth=2,
                label=inj_type.replace("_", " "))

    ax.set_title("Marker Detection Rate by Task Complexity")
    ax.set_xlabel("Task Complexity")
    ax.set_ylabel("Marker Detection Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "marker_persistence_detailed.png", dpi=150)
    plt.close()
    print("Saved: marker_persistence_detailed.png")

    # Print summary table
    print("\nMarker Detection Rate by Injection Type × Topology:")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # By complexity
    pivot_comp = df.groupby(["injection_type", "complexity"])["marker_detected"].mean().unstack()
    pivot_comp = pivot_comp.reindex(columns=complexity_order)
    print("\nMarker Detection Rate by Injection Type × Complexity:")
    print(pivot_comp.to_string(float_format=lambda x: f"{x:.3f}"))

    return df


def analyze_qualitative_examples(results):
    """Extract and analyze qualitative examples of injection propagation."""
    print("\n" + "=" * 60)
    print("QUALITATIVE PROPAGATION EXAMPLES")
    print("=" * 60)

    examples = []

    # Find most interesting examples: high propagation
    for r in results:
        if r["injection_type"] != "payload_propagation":
            continue
        if r["topology"] != "chain":
            continue
        if r["trial"] != 0:
            continue

        print(f"\n--- Example: {r['injection_type']} / {r['topology']} / {r['complexity']} ---")
        for agent_name, response in r["agent_responses"].items():
            has_marker = "xj7-propagation-active" in response.lower()
            marker_str = " [MARKER DETECTED]" if has_marker else ""
            print(f"\n{agent_name}{marker_str}:")
            print(f"  {response[:300]}...")

        examples.append({
            "injection_type": r["injection_type"],
            "topology": r["topology"],
            "complexity": r["complexity"],
            "ipr": r.get("injection_persistence_rate", 0),
        })

    return examples


def analyze_task_degradation_detailed(results):
    """Analyze task degradation patterns in more detail."""
    print("\n" + "=" * 60)
    print("TASK DEGRADATION ANALYSIS")
    print("=" * 60)

    with open(RESULTS_DIR / "task_degradation.json") as f:
        degradation = json.load(f)

    rows = []
    for d in degradation:
        for agent_name, deg_info in d["task_degradation"].items():
            rows.append({
                "injection_type": d["injection_type"],
                "topology": d["topology"],
                "complexity": d["complexity"],
                "agent": agent_name,
                "task_degradation": deg_info["task_degradation"],
                "task_similarity": deg_info["task_similarity_to_clean"],
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No degradation data available")
        return

    # Average degradation by injection type
    print("\nAverage Task Degradation by Injection Type:")
    type_deg = df.groupby("injection_type")["task_degradation"].agg(["mean", "std"])
    print(type_deg.to_string(float_format=lambda x: f"{x:.4f}"))

    # Is there significant degradation vs 0?
    for inj_type in df["injection_type"].unique():
        subset = df[df["injection_type"] == inj_type]["task_degradation"]
        t, p = stats.ttest_1samp(subset, 0)
        print(f"  {inj_type}: t={t:.3f}, p={p:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    complexity_order = ["simple", "medium", "complex"]

    ax = axes[0]
    sns.boxplot(data=df, x="injection_type", y="task_degradation", ax=ax)
    ax.set_title("Task Degradation Distribution by Injection Type")
    ax.set_xlabel("Injection Type")
    ax.set_ylabel("Task Degradation")
    ax.set_xticklabels([t.get_text().replace("_", "\n") for t in ax.get_xticklabels()],
                        fontsize=9)

    ax = axes[1]
    for topo in ["chain", "star", "mesh"]:
        subset = df[df["topology"] == topo]
        means = subset.groupby("complexity")["task_degradation"].mean()
        means = means.reindex(complexity_order)
        ax.plot(complexity_order, means.values, marker='o', linewidth=2, label=topo)

    ax.set_title("Task Degradation by Complexity and Topology")
    ax.set_xlabel("Task Complexity")
    ax.set_ylabel("Avg Task Degradation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task_degradation_detailed.png", dpi=150)
    plt.close()
    print("Saved: task_degradation_detailed.png")

    return df


def enhanced_detection_analysis(results):
    """Enhanced detection using marker presence as ground truth for injected agents."""
    print("\n" + "=" * 60)
    print("ENHANCED DETECTION ANALYSIS")
    print("=" * 60)

    # Approach: Use a combination of semantic similarity and lexical overlap
    # to detect whether an agent's output has been influenced by injection.
    #
    # Ground truth: Is the injection type != 'clean'?
    # We evaluate how well semantic sim and lexical overlap distinguish
    # injected from clean conditions.

    clean_features = []
    injected_features = []

    for r in results:
        persistence = r.get("persistence_metrics", {})
        for info in persistence.values():
            feat = {
                "semantic_sim": info["semantic_similarity"],
                "lexical_overlap": info["lexical_overlap"],
            }
            if r["injection_type"] == "clean":
                clean_features.append(feat)
            else:
                injected_features.append(feat)

    # But clean runs don't have persistence_metrics, so we need a different approach
    # Use the clean results' agent responses and compute similarity to injections
    from multi_agent_framework import get_embedding, cosine_similarity, INJECTION_TYPES

    print("Computing detection features for clean baselines...")
    clean_sims = []
    for r in results:
        if r["injection_type"] != "clean":
            continue
        for agent_name, response in r["agent_responses"].items():
            resp_emb = get_embedding(response)
            # Compute max similarity to any injection type
            max_sim = 0
            for inj_content in INJECTION_TYPES.values():
                inj_emb = get_embedding(inj_content)
                sim = cosine_similarity(resp_emb, inj_emb)
                max_sim = max(max_sim, sim)
            clean_sims.append(max_sim)

    print("Computing detection features for injected conditions...")
    injected_sims = []
    for r in results:
        if r["injection_type"] == "clean":
            continue
        persistence = r.get("persistence_metrics", {})
        for info in persistence.values():
            injected_sims.append(info["semantic_similarity"])

    clean_arr = np.array(clean_sims)
    injected_arr = np.array(injected_sims)

    print(f"\nClean similarity distribution: {clean_arr.mean():.4f} ± {clean_arr.std():.4f}")
    print(f"Injected similarity distribution: {injected_arr.mean():.4f} ± {injected_arr.std():.4f}")

    # Statistical test
    t_stat, p_val = stats.mannwhitneyu(injected_arr, clean_arr, alternative='greater')
    print(f"Mann-Whitney U test: U-statistic, p={p_val:.6f}")

    # ROC-like analysis with proper thresholds
    all_sims = np.concatenate([clean_arr, injected_arr])
    all_labels = np.concatenate([np.zeros(len(clean_arr)), np.ones(len(injected_arr))])

    thresholds = np.percentile(all_sims, np.arange(0, 101, 2))
    thresholds = np.unique(thresholds)

    roc_data = []
    for thresh in thresholds:
        predicted = all_sims >= thresh
        tp = np.sum(predicted & (all_labels == 1))
        fp = np.sum(predicted & (all_labels == 0))
        fn = np.sum(~predicted & (all_labels == 1))
        tn = np.sum(~predicted & (all_labels == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0

        roc_data.append({
            "threshold": float(thresh),
            "tpr": tpr, "fpr": fpr,
            "precision": precision, "recall": tpr, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    best_f1 = max(roc_data, key=lambda x: x["f1"])
    print(f"\nBest F1 detection:")
    print(f"  Threshold: {best_f1['threshold']:.4f}")
    print(f"  Precision: {best_f1['precision']:.3f}")
    print(f"  Recall: {best_f1['recall']:.3f}")
    print(f"  F1: {best_f1['f1']:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Distribution comparison
    ax = axes[0]
    ax.hist(clean_arr, bins=30, alpha=0.6, label=f'Clean (n={len(clean_arr)})',
            color='blue', density=True)
    ax.hist(injected_arr, bins=30, alpha=0.6, label=f'Injected (n={len(injected_arr)})',
            color='red', density=True)
    ax.axvline(x=best_f1["threshold"], color='black', linestyle='--', label=f'Best threshold')
    ax.set_title("Semantic Similarity Distributions\n(Clean vs Injected Agent Outputs)")
    ax.set_xlabel("Max Semantic Similarity to Any Injection")
    ax.set_ylabel("Density")
    ax.legend()

    # ROC curve
    ax = axes[1]
    fprs = [d["fpr"] for d in roc_data]
    tprs = [d["tpr"] for d in roc_data]
    ax.plot(fprs, tprs, 'b-', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    # AUC approximation
    sorted_pairs = sorted(zip(fprs, tprs))
    auc = np.trapezoid([p[1] for p in sorted_pairs], [p[0] for p in sorted_pairs])
    ax.set_title(f"ROC Curve (AUC ≈ {abs(auc):.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)

    # Precision-Recall
    ax = axes[2]
    recalls = [d["recall"] for d in roc_data]
    precisions = [d["precision"] for d in roc_data]
    ax.plot(recalls, precisions, 'r-', linewidth=2)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Enhanced Detection Analysis: Semantic Similarity-Based Detection", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "enhanced_detection.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: enhanced_detection.png")

    # Save detection results
    detection_results = {
        "clean_distribution": {"mean": float(clean_arr.mean()), "std": float(clean_arr.std())},
        "injected_distribution": {"mean": float(injected_arr.mean()), "std": float(injected_arr.std())},
        "mann_whitney_p": float(p_val),
        "best_f1": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in best_f1.items()},
        "auc": float(abs(auc)),
    }
    with open(RESULTS_DIR / "enhanced_detection.json", "w") as f:
        json.dump(detection_results, f, indent=2)

    return detection_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    results = load_results()
    print(f"Loaded {len(results)} experimental results")

    analyze_per_agent_propagation(results)
    marker_df = analyze_marker_persistence_detailed(results)
    examples = analyze_qualitative_examples(results)
    deg_df = analyze_task_degradation_detailed(results)
    detection = enhanced_detection_analysis(results)

    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS COMPLETE")
    print("=" * 60)
