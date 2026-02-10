"""
Figure generation for Adversarial Robustness Experiments.

Reads from adversarial_results.json and generates:
  Fig A1: Adversarial Observation Injection — posterior drift over time
  Fig A2: Adversarial Observation Injection — drift decay after injection stops
  Fig A3: Sycophancy Test — bar chart per model
  Fig A4: Prompt Injection — single-variant verdict per model
  Fig A5: Cross-Provider Robustness — heatmap (injection variant × model)
  Fig A6: Combined Adversarial Dashboard

Usage:
    python plot_adversarial_results.py                      # from saved JSON
    python -c "from plot_adversarial_results import plot_all; plot_all(results)"
"""

import os, json, sys
import numpy as np
import matplotlib
# Use non-interactive backend when running from CLI (no display)
if not os.environ.get("DISPLAY") and "inline" not in matplotlib.get_backend():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(PROJECT_DIR, "adversarial_results.json")
FIG_DIR = PROJECT_DIR  # save figures alongside data


def _load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  FIG A1 — Adversarial Observation Injection: Posterior Drift Timeline
# ═══════════════════════════════════════════════════════════════════════════

def fig_a1_drift_timeline(exp2):
    """Line plot showing clean vs adversarial posterior + injection magnitude."""
    drift = exp2["drift_analysis"]
    ts = [d["timestep"] for d in drift]
    clean = [d["clean_posterior_mw"] for d in drift]
    adv = [d["adversarial_posterior_mw"] for d in drift]
    injected = [d["injected_shift_mw"] for d in drift]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(ts, clean, "o-", color="#1565C0", linewidth=2, markersize=5,
             label="Clean Posterior (MW)")
    ax1.plot(ts, adv, "s--", color="#E53935", linewidth=2, markersize=5,
             label="Adversarial Posterior (MW)")
    ax1.set_xlabel("Timestep", fontsize=11)
    ax1.set_ylabel("Posterior Belief (MW)", fontsize=11)
    ax1.set_title("Experiment 2: Adversarial Observation Injection — Posterior Drift",
                   fontsize=13, fontweight="bold")

    # Shade injection zone
    inj_start = min(d["timestep"] for d in drift if d["injected_shift_mw"] > 0)
    inj_end = max(d["timestep"] for d in drift if d["injected_shift_mw"] > 0)
    ax1.axvspan(inj_start - 0.5, inj_end + 0.5, alpha=0.12, color="red",
                label=f"Injection zone (T{inj_start}–T{inj_end})")

    # Mark action changes
    for d in drift:
        if d["action_changed"]:
            ax1.axvline(d["timestep"], color="#FF6F00", linestyle=":", linewidth=1.5)
            ax1.annotate("Action\nChanged", xy=(d["timestep"], d["adversarial_posterior_mw"]),
                         xytext=(d["timestep"] + 1.5, d["adversarial_posterior_mw"] + 400),
                         fontsize=8, color="#FF6F00", fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color="#FF6F00"))

    # Secondary axis for injection magnitude
    ax2 = ax1.twinx()
    ax2.bar(ts, injected, alpha=0.25, color="#FF9800", width=0.6, label="Injected Shift (MW)")
    ax2.set_ylabel("Injected Shift (MW)", fontsize=11, color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")
    ax2.set_ylim(0, max(injected) * 2.5 if max(injected) > 0 else 100)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_a1_drift_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG A2 — Drift Decay After Injection Stops
# ═══════════════════════════════════════════════════════════════════════════

def fig_a2_drift_decay(exp2):
    """Bar chart showing how drift decays after injection stops."""
    drift = exp2["drift_analysis"]
    ts = [d["timestep"] for d in drift]
    drift_mw = [d["drift_mw"] for d in drift]
    injected = [d["injected_shift_mw"] for d in drift]

    # Classify bars: injection zone vs tail
    colors = []
    for d in drift:
        if d["injected_shift_mw"] > 0:
            colors.append("#E53935")  # red for injection period
        elif d["drift_mw"] > 1.0:
            colors.append("#FF9800")  # orange for residual drift
        else:
            colors.append("#4CAF50")  # green for recovered

    fig, ax = plt.subplots(figsize=(12, 4.5))
    bars = ax.bar(ts, drift_mw, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Posterior Drift (MW)", fontsize=11)
    ax.set_title("Experiment 2: Posterior Drift Magnitude & Recovery",
                  fontsize=13, fontweight="bold")

    # Annotate peak
    peak_idx = np.argmax(drift_mw)
    ax.annotate(f"Peak: +{drift_mw[peak_idx]:.0f} MW",
                xy=(ts[peak_idx], drift_mw[peak_idx]),
                xytext=(ts[peak_idx] + 2, drift_mw[peak_idx] + 30),
                fontsize=9, fontweight="bold", color="#E53935",
                arrowprops=dict(arrowstyle="->", color="#E53935"))

    # Annotate recovery point (first timestep where drift < 5 MW after injection)
    inj_end_idx = max(i for i, d in enumerate(drift) if d["injected_shift_mw"] > 0)
    for i in range(inj_end_idx + 1, len(drift)):
        if abs(drift_mw[i]) < 5.0:
            ax.annotate(f"Recovered\n(<5 MW @ T{ts[i]})",
                        xy=(ts[i], drift_mw[i]),
                        xytext=(ts[i] + 1, max(drift_mw) * 0.3),
                        fontsize=8, fontweight="bold", color="#4CAF50",
                        arrowprops=dict(arrowstyle="->", color="#4CAF50"))
            break

    # Legend
    patches = [
        mpatches.Patch(color="#E53935", label="Active injection"),
        mpatches.Patch(color="#FF9800", label="Residual drift"),
        mpatches.Patch(color="#4CAF50", label="Recovered (<5 MW)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_a2_drift_decay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG A3 — Sycophancy Test: Per-Model Rates
# ═══════════════════════════════════════════════════════════════════════════

def fig_a3_sycophancy(exp1):
    """Stacked bar chart: sycophantic vs flagged vs API error per model."""
    models = [r["model"] for r in exp1]
    total = [r["total_injected"] for r in exp1]
    syc = [r["sycophantic_count"] for r in exp1]
    flagged = [r["flagged_count"] for r in exp1]
    api_err = [t - s - f for t, s, f in zip(total, syc, flagged)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.5

    bars_syc = ax.bar(x, syc, width, label="Sycophantic (rationalized)", color="#E53935")
    bars_flag = ax.bar(x, flagged, width, bottom=syc, label="Flagged as wrong", color="#4CAF50")
    if any(e > 0 for e in api_err):
        bars_err = ax.bar(x, api_err, width,
                          bottom=[s + f for s, f in zip(syc, flagged)],
                          label="API Error", color="#9E9E9E")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Count (out of 20)", fontsize=11)
    ax.set_title("Experiment 1: Sycophancy Test — Do LLMs Rationalize Wrong Actions?",
                  fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(total) + 3)
    ax.legend(fontsize=10)

    # Annotate rates
    for i, (s, f, t) in enumerate(zip(syc, flagged, total)):
        valid = s + f
        if valid > 0:
            rate = s / valid * 100
            ax.text(i, t + 0.5, f"{rate:.0f}% syc.", ha="center",
                    fontweight="bold", fontsize=10,
                    color="#E53935" if rate > 50 else "#4CAF50")

    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_a3_sycophancy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG A4 — Prompt Injection: Single Variant Verdict
# ═══════════════════════════════════════════════════════════════════════════

def fig_a4_prompt_injection(exp3):
    """Horizontal bar chart showing prompt injection verdict per model."""
    models = []
    verdicts = []
    colors = []
    color_map = {
        "VULNERABLE": "#E53935",
        "ROBUST (flagged)": "#1565C0",
        "ROBUST (ignored injection)": "#4CAF50",
        "API_ERROR": "#9E9E9E",
    }
    verdict_scores = {
        "VULNERABLE": 0,
        "ROBUST (ignored injection)": 1,
        "ROBUST (flagged)": 2,
        "API_ERROR": -1,
    }

    for r in exp3:
        models.append(r["model"])
        v = r["verdict"]
        verdicts.append(v)
        colors.append(color_map.get(v, "#9E9E9E"))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    y = np.arange(len(models))
    scores = [verdict_scores.get(v, 0) for v in verdicts]
    bars = ax.barh(y, [max(0, s) for s in scores], color=colors, height=0.5,
                   edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Vulnerable", "Ignored\nInjection", "Flagged\nInjection"], fontsize=9)
    ax.set_xlim(-0.5, 2.5)
    ax.set_title('Experiment 3: Prompt Injection — "IGNORE ALL PREVIOUS INSTRUCTIONS"',
                  fontsize=12, fontweight="bold")

    # Annotate verdict text on bars
    for i, (v, s) in enumerate(zip(verdicts, scores)):
        label = v.replace("ROBUST ", "").replace("(", "").replace(")", "")
        x_pos = max(0, s) + 0.05
        ax.text(x_pos, i, label, va="center", fontsize=10, fontweight="bold",
                color=colors[i])

    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_a4_prompt_injection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG A5 — Cross-Provider Robustness Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig_a5_cross_provider_heatmap(exp4):
    """Heatmap: injection variant (rows) × model (cols), colored by verdict."""
    variants_data = exp4["injection_variants"]
    variant_names = list(variants_data.keys())

    # Collect model names from the first variant
    model_names = [r["model"] for r in variants_data[variant_names[0]]]

    # Build verdict matrix: 0=VULN, 1=PARTIAL, 2=ROBUST(ignored), 3=ROBUST(flagged), -1=ERR
    verdict_map = {
        "VULNERABLE": 0,
        "PARTIALLY ROBUST": 1,
        "ROBUST (ignored)": 2,
        "ROBUST (flagged)": 3,
        "API_ERROR": -1,
    }
    cmap_colors = ["#9E9E9E", "#E53935", "#FFB74D", "#66BB6A", "#1565C0"]
    cmap = ListedColormap(cmap_colors)

    matrix = np.zeros((len(variant_names), len(model_names)), dtype=int)
    text_matrix = []

    for i, vname in enumerate(variant_names):
        row_texts = []
        for j, r in enumerate(variants_data[vname]):
            v = r["verdict"]
            matrix[i, j] = verdict_map.get(v, -1) + 1  # shift so -1 → 0, 0 → 1, etc.
            short = {
                "VULNERABLE": "VULN", "PARTIALLY ROBUST": "PARTIAL",
                "ROBUST (ignored)": "OK", "ROBUST (flagged)": "FLAG",
                "API_ERROR": "ERR",
            }.get(v, "?")
            row_texts.append(short)
        text_matrix.append(row_texts)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=4)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(variant_names)))
    nice_names = [n.replace("_", " ").title() for n in variant_names]
    ax.set_yticklabels(nice_names, fontsize=10)
    ax.set_title("Experiment 4: Cross-Provider Prompt Injection Robustness",
                  fontsize=13, fontweight="bold", pad=15)

    # Annotate cells
    for i in range(len(variant_names)):
        for j in range(len(model_names)):
            text = text_matrix[i][j]
            text_color = "white" if matrix[i, j] in [0, 1, 4] else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    fontweight="bold", color=text_color)

    # Legend
    patches = [
        mpatches.Patch(color=cmap_colors[0], label="API Error"),
        mpatches.Patch(color=cmap_colors[1], label="Vulnerable"),
        mpatches.Patch(color=cmap_colors[2], label="Partially Robust"),
        mpatches.Patch(color=cmap_colors[3], label="Robust (ignored)"),
        mpatches.Patch(color=cmap_colors[4], label="Robust (flagged)"),
    ]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=5, fontsize=8)

    # Vuln rates at bottom
    vuln_rates = exp4.get("vulnerability_rates", {})
    if vuln_rates:
        for j, m in enumerate(model_names):
            rate = vuln_rates.get(m, 0)
            ax.text(j, len(variant_names) - 0.3, f"{rate:.0f}% vuln",
                    ha="center", va="top", fontsize=8, color="#E53935",
                    fontweight="bold")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_a5_cross_provider_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG A6 — Combined Adversarial Dashboard
# ═══════════════════════════════════════════════════════════════════════════

def fig_a6_dashboard(results):
    """4-panel dashboard summarizing all experiments."""
    has_exp1 = "experiment_1_sycophancy" in results
    has_exp3 = "experiment_3_prompt_injection" in results
    has_exp4 = "experiment_4_cross_provider" in results
    exp2 = results["experiment_2_adversarial_obs"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adversarial Robustness Dashboard", fontsize=15, fontweight="bold", y=0.98)

    # ── Panel 1 (top-left): Drift timeline ──
    ax = axes[0, 0]
    drift = exp2["drift_analysis"]
    ts = [d["timestep"] for d in drift]
    drift_mw = [d["drift_mw"] for d in drift]
    injected = [d["injected_shift_mw"] for d in drift]

    ax.bar(ts, injected, alpha=0.3, color="#FF9800", label="Injected (MW)")
    ax.plot(ts, drift_mw, "o-", color="#E53935", linewidth=2, markersize=4,
            label="Posterior Drift (MW)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("MW")
    ax.set_title(f"Exp 2: Obs Injection (peak drift: +{exp2['peak_drift_mw']:.0f} MW)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2 (top-right): Sycophancy ──
    ax = axes[0, 1]
    if has_exp1:
        exp1 = results["experiment_1_sycophancy"]
        models = [r["model"] for r in exp1]
        syc = [r["sycophancy_rate"] for r in exp1]
        flag = [r["flag_rate"] for r in exp1]
        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w / 2, syc, w, label="Sycophantic %", color="#E53935")
        ax.bar(x + w / 2, flag, w, label="Flagged %", color="#4CAF50")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Rate (%)")
        ax.set_title("Exp 1: Sycophancy Rates")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        for i in range(len(models)):
            ax.text(i - w / 2, syc[i] + 1, f"{syc[i]:.0f}%", ha="center", fontsize=9,
                    fontweight="bold", color="#E53935")
            ax.text(i + w / 2, flag[i] + 1, f"{flag[i]:.0f}%", ha="center", fontsize=9,
                    fontweight="bold", color="#4CAF50")
    else:
        ax.text(0.5, 0.5, "Exp 1: Sycophancy\n(API key required)",
                ha="center", va="center", fontsize=12, color="#9E9E9E",
                transform=ax.transAxes)
        ax.set_title("Exp 1: Sycophancy Rates (skipped)")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3 (bottom-left): Prompt Injection ──
    ax = axes[1, 0]
    if has_exp3:
        exp3 = results["experiment_3_prompt_injection"]
        models = [r["model"] for r in exp3]
        verdict_labels = [r["verdict"] for r in exp3]
        color_map = {
            "VULNERABLE": "#E53935", "ROBUST (flagged)": "#1565C0",
            "ROBUST (ignored injection)": "#4CAF50", "API_ERROR": "#9E9E9E",
        }
        bar_colors = [color_map.get(v, "#9E9E9E") for v in verdict_labels]
        # Map to numeric for bar height: 2=flagged, 1=ignored, 0=vuln
        height_map = {"VULNERABLE": 0.3, "ROBUST (ignored injection)": 0.7,
                      "ROBUST (flagged)": 1.0, "API_ERROR": 0.1}
        heights = [height_map.get(v, 0.1) for v in verdict_labels]

        ax.bar(models, heights, color=bar_colors, edgecolor="white", width=0.5)
        for i, (m, v) in enumerate(zip(models, verdict_labels)):
            short = v.replace("ROBUST ", "").replace("(", "").replace(")", "")
            ax.text(i, heights[i] + 0.03, short, ha="center", fontsize=9,
                    fontweight="bold", color=bar_colors[i])
        ax.set_ylim(0, 1.3)
        ax.set_ylabel("Robustness")
        ax.set_yticks([0, 0.3, 0.7, 1.0])
        ax.set_yticklabels(["", "Vulnerable", "Ignored", "Flagged"], fontsize=8)
        ax.set_title("Exp 3: Prompt Injection Verdict")
    else:
        ax.text(0.5, 0.5, "Exp 3: Prompt Injection\n(API key required)",
                ha="center", va="center", fontsize=12, color="#9E9E9E",
                transform=ax.transAxes)
        ax.set_title("Exp 3: Prompt Injection (skipped)")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4 (bottom-right): Cross-Provider Vulnerability Rates ──
    ax = axes[1, 1]
    if has_exp4:
        exp4 = results["experiment_4_cross_provider"]
        vuln_rates = exp4["vulnerability_rates"]
        models = list(vuln_rates.keys())
        rates = list(vuln_rates.values())
        bar_colors = ["#E53935" if r > 0 else "#4CAF50" for r in rates]
        ax.bar(models, rates, color=bar_colors, width=0.5, edgecolor="white")
        for i, r in enumerate(rates):
            ax.text(i, r + 1, f"{r:.0f}%", ha="center", fontsize=11,
                    fontweight="bold", color=bar_colors[i])
        ax.set_ylabel("Vulnerability Rate (%)")
        ax.set_ylim(0, max(rates) + 20 if max(rates) > 0 else 30)
        ax.set_title("Exp 4: Cross-Provider Vulnerability (5 injection types)")
    else:
        ax.text(0.5, 0.5, "Exp 4: Cross-Provider\n(API key required)",
                ha="center", va="center", fontsize=12, color="#9E9E9E",
                transform=ax.transAxes)
        ax.set_title("Exp 4: Cross-Provider Robustness (skipped)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIG_DIR, "fig_a6_adversarial_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def plot_all(results=None):
    """Generate all adversarial figures from results dict or saved JSON."""
    if results is None:
        print(f"Loading results from {RESULTS_PATH}...")
        results = _load_results()

    print("\nGenerating adversarial experiment figures...\n")

    # Exp 2 figures (always available — computational)
    exp2 = results.get("experiment_2_adversarial_obs")
    if exp2 and exp2.get("drift_analysis"):
        fig_a1_drift_timeline(exp2)
        fig_a2_drift_decay(exp2)
    else:
        print("  Skipping Fig A1/A2 — no Experiment 2 data.")

    # Exp 1 figure (needs API)
    exp1 = results.get("experiment_1_sycophancy")
    if exp1 and any(r.get("flagged_count", 0) + r.get("sycophantic_count", 0) > 0 for r in exp1):
        fig_a3_sycophancy(exp1)
    else:
        print("  Skipping Fig A3 — no valid Experiment 1 data (API key needed).")

    # Exp 3 figure (needs API)
    exp3 = results.get("experiment_3_prompt_injection")
    if exp3 and any(not r.get("api_error") for r in exp3):
        fig_a4_prompt_injection(exp3)
    else:
        print("  Skipping Fig A4 — no valid Experiment 3 data (API key needed).")

    # Exp 4 figure (needs API)
    exp4 = results.get("experiment_4_cross_provider")
    if exp4 and exp4.get("injection_variants"):
        # Check if any variant has real (non-error) results
        has_real = any(
            any(not r.get("api_error") for r in variant_results)
            for variant_results in exp4["injection_variants"].values()
        )
        if has_real:
            fig_a5_cross_provider_heatmap(exp4)
        else:
            print("  Skipping Fig A5 — all Experiment 4 results are API errors.")
    else:
        print("  Skipping Fig A5 — no Experiment 4 data (API key needed).")

    # Dashboard (always generated, shows placeholders for missing data)
    fig_a6_dashboard(results)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    plot_all()
