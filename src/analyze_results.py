"""
Analysis and visualization for adversarial prompt propagation experiments.

Produces statistical tests, visualizations, and a summary of findings.
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results" / "data"
PLOTS_DIR = Path(__file__).parent.parent / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def load_results():
    """Load experiment results."""
    with open(RESULTS_DIR / "experiment_results.json") as f:
        results = json.load(f)
    return results


def load_degradation():
    """Load task degradation results."""
    with open(RESULTS_DIR / "task_degradation.json") as f:
        return json.load(f)


def build_dataframe(results):
    """Build a pandas DataFrame from results for easier analysis."""
    rows = []
    for r in results:
        row = {
            "topology": r["topology"],
            "complexity": r["complexity"],
            "injection_type": r["injection_type"],
            "trial": r["trial"],
            "n_messages": r["n_messages"],
            "elapsed_seconds": r["elapsed_seconds"],
        }

        if r["injection_type"] != "clean":
            row["injection_persistence_rate"] = r.get("injection_persistence_rate", 0)
            row["avg_semantic_similarity"] = r.get("avg_semantic_similarity", 0)
            row["any_marker_detected"] = r.get("any_marker_detected", False)

            # Extract per-agent persistence info
            persistence = r.get("persistence_metrics", {})
            sem_sims = [info["semantic_similarity"] for info in persistence.values()]
            lex_overlaps = [info["lexical_overlap"] for info in persistence.values()]

            row["max_semantic_sim"] = max(sem_sims) if sem_sims else 0
            row["min_semantic_sim"] = min(sem_sims) if sem_sims else 0
            row["std_semantic_sim"] = np.std(sem_sims) if sem_sims else 0
            row["avg_lexical_overlap"] = np.mean(lex_overlaps) if lex_overlaps else 0

            # Mutation metrics
            mutation = r.get("mutation_metrics", [])
            if mutation:
                row["initial_hop_sim"] = mutation[0]["semantic_similarity_to_injection"]
                row["final_hop_sim"] = mutation[-1]["semantic_similarity_to_injection"]
                row["sim_decay"] = mutation[0]["semantic_similarity_to_injection"] - mutation[-1]["semantic_similarity_to_injection"]
                row["avg_bigram_entropy"] = np.mean([m["bigram_entropy"] for m in mutation])
            else:
                row["initial_hop_sim"] = 0
                row["final_hop_sim"] = 0
                row["sim_decay"] = 0
                row["avg_bigram_entropy"] = 0
        else:
            row["injection_persistence_rate"] = None
            row["avg_semantic_similarity"] = None
            row["any_marker_detected"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def plot_persistence_by_topology(df):
    """Plot injection persistence rate by topology."""
    injected = df[df["injection_type"] != "clean"].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=injected, x="topology", y="avg_semantic_similarity",
                hue="injection_type", ax=ax)
    ax.set_title("Semantic Similarity to Injection by Topology and Injection Type")
    ax.set_xlabel("Network Topology")
    ax.set_ylabel("Average Semantic Similarity to Injection")
    ax.legend(title="Injection Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "persistence_by_topology.png", dpi=150)
    plt.close()
    print("Saved: persistence_by_topology.png")


def plot_persistence_by_complexity(df):
    """Plot injection persistence rate by task complexity."""
    injected = df[df["injection_type"] != "clean"].copy()

    # Order complexity levels
    complexity_order = ["simple", "medium", "complex"]
    injected["complexity"] = pd.Categorical(injected["complexity"], categories=complexity_order, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Semantic similarity
    sns.boxplot(data=injected, x="complexity", y="avg_semantic_similarity",
                hue="topology", ax=axes[0])
    axes[0].set_title("Semantic Similarity to Injection\nby Task Complexity")
    axes[0].set_xlabel("Task Complexity")
    axes[0].set_ylabel("Avg Semantic Similarity")

    # Marker persistence rate
    sns.boxplot(data=injected, x="complexity", y="injection_persistence_rate",
                hue="topology", ax=axes[1])
    axes[1].set_title("Injection Persistence Rate\nby Task Complexity")
    axes[1].set_xlabel("Task Complexity")
    axes[1].set_ylabel("Persistence Rate")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "persistence_by_complexity.png", dpi=150)
    plt.close()
    print("Saved: persistence_by_complexity.png")


def plot_mutation_across_hops(results):
    """Plot how semantic similarity to injection decays across hops."""
    hop_data = defaultdict(lambda: defaultdict(list))

    for r in results:
        if r["injection_type"] == "clean":
            continue
        mutation = r.get("mutation_metrics", [])
        for m in mutation:
            key = (r["topology"], r["injection_type"])
            hop_data[key][m["hop"]].append(m["semantic_similarity_to_injection"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    topologies = ["chain", "star", "mesh"]

    for idx, topo in enumerate(topologies):
        ax = axes[idx]
        for inj_type in ["direct_override", "context_poisoning", "role_hijack",
                          "payload_propagation", "subtle_bias"]:
            key = (topo, inj_type)
            if key in hop_data:
                hops = sorted(hop_data[key].keys())
                means = [np.mean(hop_data[key][h]) for h in hops]
                stds = [np.std(hop_data[key][h]) for h in hops]
                ax.errorbar(hops, means, yerr=stds, marker='o', label=inj_type.replace("_", " "),
                           capsize=3, linewidth=2)

        ax.set_title(f"Topology: {topo.title()}")
        ax.set_xlabel("Hop Number")
        ax.set_ylabel("Semantic Similarity to Injection")
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Adversarial Content Semantic Drift Across Agent Hops", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mutation_across_hops.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: mutation_across_hops.png")


def plot_task_degradation(degradation_results):
    """Plot task accuracy degradation."""
    rows = []
    for d in degradation_results:
        rows.append({
            "topology": d["topology"],
            "complexity": d["complexity"],
            "injection_type": d["injection_type"],
            "avg_degradation": d["avg_degradation"],
        })
    df = pd.DataFrame(rows)

    if df.empty:
        print("No degradation data to plot")
        return

    complexity_order = ["simple", "medium", "complex"]
    df["complexity"] = pd.Categorical(df["complexity"], categories=complexity_order, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="injection_type", y="avg_degradation",
                hue="complexity", ax=ax, ci="sd")
    ax.set_title("Task Accuracy Degradation by Injection Type and Task Complexity")
    ax.set_xlabel("Injection Type")
    ax.set_ylabel("Task Degradation (1 - cosine similarity to clean)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task_degradation.png", dpi=150)
    plt.close()
    print("Saved: task_degradation.png")


def plot_injection_heatmap(df):
    """Create heatmap of injection effectiveness across all conditions."""
    injected = df[df["injection_type"] != "clean"].copy()

    pivot = injected.groupby(["injection_type", "topology"])["avg_semantic_similarity"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                vmin=0.6, vmax=0.9)
    ax.set_title("Average Semantic Similarity to Injection\n(Injection Type × Topology)")
    ax.set_xlabel("Topology")
    ax.set_ylabel("Injection Type")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "injection_heatmap.png", dpi=150)
    plt.close()
    print("Saved: injection_heatmap.png")


def plot_amplification_factor(df):
    """Plot amplification factor: how injection effect changes with topology density."""
    injected = df[df["injection_type"] != "clean"].copy()

    # Calculate amplification per injection type
    fig, ax = plt.subplots(figsize=(10, 6))

    for inj_type in injected["injection_type"].unique():
        subset = injected[injected["injection_type"] == inj_type]
        topo_means = subset.groupby("topology")["avg_semantic_similarity"].mean()

        # Order: chain < star < mesh (increasing connectivity)
        topo_order = ["chain", "star", "mesh"]
        vals = [topo_means.get(t, 0) for t in topo_order]

        ax.plot(topo_order, vals, marker='o', linewidth=2,
                label=inj_type.replace("_", " "))

    ax.set_title("Injection Semantic Persistence by Network Connectivity")
    ax.set_xlabel("Topology (increasing connectivity →)")
    ax.set_ylabel("Avg Semantic Similarity to Injection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "amplification_by_connectivity.png", dpi=150)
    plt.close()
    print("Saved: amplification_by_connectivity.png")


def statistical_tests(df):
    """Run statistical tests for hypothesis testing."""
    injected = df[df["injection_type"] != "clean"].copy()
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    stats_results = {}

    # H1: Do injections persist beyond hop 0?
    print("\n--- H1: Injection Persistence ---")
    sem_sims = injected["avg_semantic_similarity"].dropna()
    # Baseline: random text typically has ~0.65-0.70 cosine similarity
    baseline_sim = 0.70
    t_stat, p_val = stats.ttest_1samp(sem_sims, baseline_sim)
    cohens_d = (sem_sims.mean() - baseline_sim) / sem_sims.std()
    print(f"Mean semantic similarity: {sem_sims.mean():.4f} ± {sem_sims.std():.4f}")
    print(f"vs baseline {baseline_sim}: t={t_stat:.3f}, p={p_val:.6f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    stats_results["H1_persistence"] = {
        "mean": float(sem_sims.mean()),
        "std": float(sem_sims.std()),
        "baseline": baseline_sim,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "significant": p_val < 0.05,
    }

    # H3: Topology effects (one-way ANOVA on semantic similarity)
    print("\n--- H3: Topology Effects ---")
    groups = [group["avg_semantic_similarity"].dropna().values
              for _, group in injected.groupby("topology")]
    if all(len(g) > 1 for g in groups):
        f_stat, p_val = stats.f_oneway(*groups)
        # Effect size: eta-squared
        ss_between = sum(len(g) * (np.mean(g) - sem_sims.mean())**2 for g in groups)
        ss_total = sum(np.sum((g - sem_sims.mean())**2) for g in groups)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        for topo in ["chain", "star", "mesh"]:
            topo_data = injected[injected["topology"] == topo]["avg_semantic_similarity"]
            print(f"  {topo}: {topo_data.mean():.4f} ± {topo_data.std():.4f}")

        print(f"ANOVA: F={f_stat:.3f}, p={p_val:.6f}")
        print(f"η²: {eta_sq:.4f}")

        stats_results["H3_topology"] = {
            "F_stat": float(f_stat),
            "p_value": float(p_val),
            "eta_squared": float(eta_sq),
            "significant": p_val < 0.05,
            "topology_means": {
                topo: float(injected[injected["topology"] == topo]["avg_semantic_similarity"].mean())
                for topo in ["chain", "star", "mesh"]
            }
        }

    # H4: Task complexity correlation
    print("\n--- H4: Task Complexity Correlation ---")
    complexity_map = {"simple": 1, "medium": 2, "complex": 3}
    injected_copy = injected.copy()
    injected_copy["complexity_numeric"] = injected_copy["complexity"].map(complexity_map)

    sem_sim_vals = injected_copy["avg_semantic_similarity"].dropna()
    complexity_vals = injected_copy.loc[sem_sim_vals.index, "complexity_numeric"]

    rho, p_val = stats.spearmanr(complexity_vals, sem_sim_vals)
    print(f"Spearman ρ (complexity vs semantic similarity): {rho:.4f}, p={p_val:.6f}")

    for comp in ["simple", "medium", "complex"]:
        comp_data = injected[injected["complexity"] == comp]["avg_semantic_similarity"]
        print(f"  {comp}: {comp_data.mean():.4f} ± {comp_data.std():.4f}")

    stats_results["H4_complexity"] = {
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
        "complexity_means": {
            comp: float(injected[injected["complexity"] == comp]["avg_semantic_similarity"].mean())
            for comp in ["simple", "medium", "complex"]
        }
    }

    # Persistence rate analysis
    print("\n--- Injection Persistence Rate (Marker Detection) ---")
    ipr_vals = injected["injection_persistence_rate"].dropna()
    print(f"Overall IPR: {ipr_vals.mean():.4f} ± {ipr_vals.std():.4f}")

    for inj_type in injected["injection_type"].unique():
        type_data = injected[injected["injection_type"] == inj_type]["injection_persistence_rate"]
        print(f"  {inj_type}: {type_data.mean():.4f} ± {type_data.std():.4f}")

    stats_results["persistence_rate"] = {
        "overall_mean": float(ipr_vals.mean()),
        "overall_std": float(ipr_vals.std()),
        "by_type": {
            inj_type: float(injected[injected["injection_type"] == inj_type]["injection_persistence_rate"].mean())
            for inj_type in injected["injection_type"].unique()
        }
    }

    # Injection type comparison
    print("\n--- Injection Type Comparison (Kruskal-Wallis) ---")
    type_groups = [group["avg_semantic_similarity"].dropna().values
                   for _, group in injected.groupby("injection_type")]
    if all(len(g) > 1 for g in type_groups):
        h_stat, p_val = stats.kruskal(*type_groups)
        print(f"Kruskal-Wallis H={h_stat:.3f}, p={p_val:.6f}")
        stats_results["injection_type_comparison"] = {
            "H_stat": float(h_stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
        }

    # Semantic drift analysis
    print("\n--- H2: Semantic Mutation/Drift ---")
    decay_vals = injected["sim_decay"].dropna()
    print(f"Average similarity decay (hop 0 to final): {decay_vals.mean():.4f} ± {decay_vals.std():.4f}")
    t_stat, p_val = stats.ttest_1samp(decay_vals, 0)
    print(f"One-sample t-test (decay > 0): t={t_stat:.3f}, p={p_val:.6f}")
    stats_results["H2_mutation"] = {
        "mean_decay": float(decay_vals.mean()),
        "std_decay": float(decay_vals.std()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
    }

    # Save all stats
    def json_default(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, bool):
            return bool(obj)
        return str(obj)

    with open(RESULTS_DIR / "statistical_results.json", "w") as f:
        json.dump(stats_results, f, indent=2, default=json_default)
    print(f"\nStatistical results saved to {RESULTS_DIR / 'statistical_results.json'}")

    return stats_results


def detection_analysis(results):
    """Analyze detection feasibility at communication interfaces."""
    print("\n" + "=" * 60)
    print("DETECTION ANALYSIS (H5)")
    print("=" * 60)

    # For detection, we use semantic similarity thresholds
    # Compare injected vs clean agent outputs
    clean_sims = []
    injected_sims = []

    for r in results:
        persistence = r.get("persistence_metrics", {})
        for info in persistence.values():
            sim = info["semantic_similarity"]
            if r["injection_type"] == "clean":
                clean_sims.append(sim)
            else:
                injected_sims.append(sim)

    # Also collect from the semantic similarity field
    for r in results:
        if r["injection_type"] == "clean":
            # For clean runs, compute similarity to a random injection
            pass  # We'll use marker detection as ground truth instead

    # Detection based on marker presence
    # True positive: marker detected in injected run
    # False positive: marker detected in clean run (shouldn't happen)
    tp, fp, fn, tn = 0, 0, 0, 0

    thresholds_results = []
    for threshold in np.arange(0.65, 0.95, 0.02):
        tp = fp = fn = tn = 0
        for r in results:
            persistence = r.get("persistence_metrics", {})
            for info in persistence.values():
                sim = info["semantic_similarity"]
                is_injected = r["injection_type"] != "clean"
                predicted_injected = sim > threshold

                if is_injected and predicted_injected:
                    tp += 1
                elif not is_injected and predicted_injected:
                    fp += 1
                elif is_injected and not predicted_injected:
                    fn += 1
                else:
                    tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        thresholds_results.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    # Find best F1
    best = max(thresholds_results, key=lambda x: x["f1"])
    print(f"\nBest detection threshold: {best['threshold']:.2f}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall: {best['recall']:.3f}")
    print(f"  F1: {best['f1']:.3f}")
    print(f"  TP={best['tp']}, FP={best['fp']}, FN={best['fn']}, TN={best['tn']}")

    # Plot precision-recall curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    thresholds = [t["threshold"] for t in thresholds_results]
    precisions = [t["precision"] for t in thresholds_results]
    recalls = [t["recall"] for t in thresholds_results]
    f1s = [t["f1"] for t in thresholds_results]

    axes[0].plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2)
    axes[0].plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2)
    axes[0].plot(thresholds, f1s, 'g-^', label='F1', linewidth=2)
    axes[0].axvline(x=best["threshold"], color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title("Detection Performance vs Threshold")
    axes[0].set_xlabel("Semantic Similarity Threshold")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precision-recall curve
    axes[1].plot(recalls, precisions, 'k-o', linewidth=2)
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "detection_analysis.png", dpi=150)
    plt.close()
    print("Saved: detection_analysis.png")

    # Save detection results
    with open(RESULTS_DIR / "detection_results.json", "w") as f:
        json.dump({
            "best_threshold": best,
            "all_thresholds": thresholds_results,
        }, f, indent=2)

    return best, thresholds_results


def plot_summary_dashboard(df, stats_results, detection_best):
    """Create a comprehensive summary dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    injected = df[df["injection_type"] != "clean"].copy()
    complexity_order = ["simple", "medium", "complex"]

    # 1. Persistence by injection type
    ax = axes[0, 0]
    type_means = injected.groupby("injection_type")["avg_semantic_similarity"].agg(["mean", "std"])
    type_means = type_means.sort_values("mean", ascending=False)
    colors = sns.color_palette("husl", len(type_means))
    bars = ax.bar(range(len(type_means)), type_means["mean"], yerr=type_means["std"],
                  capsize=5, color=colors)
    ax.set_xticks(range(len(type_means)))
    ax.set_xticklabels([n.replace("_", "\n") for n in type_means.index], fontsize=9)
    ax.set_title("Semantic Persistence\nby Injection Type")
    ax.set_ylabel("Avg Semantic Similarity")
    ax.set_ylim(0.5, 1.0)

    # 2. Persistence by topology
    ax = axes[0, 1]
    for topo in ["chain", "star", "mesh"]:
        topo_data = injected[injected["topology"] == topo]
        means = topo_data.groupby("complexity")["avg_semantic_similarity"].mean()
        means = means.reindex(complexity_order)
        ax.plot(complexity_order, means.values, marker='o', linewidth=2, label=topo)
    ax.set_title("Semantic Persistence\nby Topology × Complexity")
    ax.set_xlabel("Task Complexity")
    ax.set_ylabel("Avg Semantic Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Marker persistence rate
    ax = axes[0, 2]
    ipr_by_type = injected.groupby("injection_type")["injection_persistence_rate"].agg(["mean", "std"])
    ipr_by_type = ipr_by_type.sort_values("mean", ascending=False)
    ax.barh(range(len(ipr_by_type)), ipr_by_type["mean"],
            xerr=ipr_by_type["std"], capsize=3,
            color=sns.color_palette("Set2", len(ipr_by_type)))
    ax.set_yticks(range(len(ipr_by_type)))
    ax.set_yticklabels([n.replace("_", " ") for n in ipr_by_type.index], fontsize=9)
    ax.set_title("Marker Persistence Rate\nby Injection Type")
    ax.set_xlabel("Persistence Rate (fraction of agents)")

    # 4. Semantic drift across hops
    ax = axes[1, 0]
    for topo in ["chain", "star", "mesh"]:
        topo_results = injected[injected["topology"] == topo]
        if not topo_results.empty:
            ax.scatter(topo_results["initial_hop_sim"], topo_results["final_hop_sim"],
                      label=topo, alpha=0.6, s=50)
    ax.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3, label='No decay')
    ax.set_title("Semantic Drift:\nInitial vs Final Hop Similarity")
    ax.set_xlabel("Initial Hop Similarity")
    ax.set_ylabel("Final Hop Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Detection performance
    ax = axes[1, 1]
    det = detection_best
    metrics = ["precision", "recall", "f1"]
    values = [det[m] for m in metrics]
    bars = ax.bar(metrics, values, color=["#2196F3", "#4CAF50", "#FF9800"])
    ax.set_title(f"Detection Performance\n(threshold={det['threshold']:.2f})")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha='center', fontsize=11, fontweight='bold')

    # 6. Summary statistics text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = "KEY FINDINGS\n" + "=" * 30 + "\n\n"

    h1 = stats_results.get("H1_persistence", {})
    summary_text += f"H1 (Persistence):\n"
    summary_text += f"  Mean sim: {h1.get('mean', 0):.3f}\n"
    summary_text += f"  p = {h1.get('p_value', 1):.4f}\n"
    summary_text += f"  {'✓ Significant' if h1.get('significant') else '✗ Not significant'}\n\n"

    h3 = stats_results.get("H3_topology", {})
    summary_text += f"H3 (Topology):\n"
    summary_text += f"  F = {h3.get('F_stat', 0):.3f}\n"
    summary_text += f"  p = {h3.get('p_value', 1):.4f}\n"
    summary_text += f"  {'✓ Significant' if h3.get('significant') else '✗ Not significant'}\n\n"

    h4 = stats_results.get("H4_complexity", {})
    summary_text += f"H4 (Complexity):\n"
    summary_text += f"  ρ = {h4.get('spearman_rho', 0):.3f}\n"
    summary_text += f"  p = {h4.get('p_value', 1):.4f}\n"
    summary_text += f"  {'✓ Significant' if h4.get('significant') else '✗ Not significant'}\n\n"

    summary_text += f"H5 (Detection):\n"
    summary_text += f"  Precision: {det['precision']:.3f}\n"
    summary_text += f"  Recall: {det['recall']:.3f}\n"
    summary_text += f"  F1: {det['f1']:.3f}\n"

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle("Adversarial Prompt Propagation: Experimental Results Summary",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: summary_dashboard.png")


if __name__ == "__main__":
    print("Loading results...")
    results = load_results()
    df = build_dataframe(results)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Conditions: {df['injection_type'].value_counts().to_dict()}")

    # Generate all plots
    print("\nGenerating plots...")
    plot_persistence_by_topology(df)
    plot_persistence_by_complexity(df)
    plot_mutation_across_hops(results)
    plot_injection_heatmap(df)
    plot_amplification_factor(df)

    # Load and plot degradation
    try:
        degradation = load_degradation()
        plot_task_degradation(degradation)
    except Exception as e:
        print(f"Warning: Could not plot degradation: {e}")

    # Statistical tests
    stats_results = statistical_tests(df)

    # Detection analysis
    detection_best, _ = detection_analysis(results)

    # Summary dashboard
    plot_summary_dashboard(df, stats_results, detection_best)

    print("\n✓ Analysis complete! All plots saved to results/plots/")
