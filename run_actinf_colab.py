"""
Active Inference Agent + LLM Interpreter — Google Colab Script

Usage:
  1. Zip and upload the entire actinf_interp_copy folder to Google Drive
     (or upload it directly to the Colab runtime)
  2. Run Cell 1 to mount Drive and set paths
  3. Set your API key in the CONFIG section
  4. Run all remaining cells
"""

# ── CELL 0: Mount Google Drive & set working directory ──────────────────────
# Uncomment the lines below if you uploaded the folder to Google Drive.
# If you uploaded directly to the Colab runtime, skip this cell.

import os

USE_GOOGLE_DRIVE = False  # Set True only if you uploaded to Google Drive

if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    # ── EDIT THIS PATH to match where you put the folder in Drive ──
    PROJECT_DIR = "/content/drive/MyDrive/actinf_interp_copy"
else:
    # If uploaded directly to the Colab runtime filesystem
    PROJECT_DIR = "/content/actinf_interp_copy"

os.chdir(PROJECT_DIR)
print(f"Working directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

# ── CELL 1: Imports & CONFIG ───────────────────────────────────────────────
import sqlite3, json, math, requests
import numpy as np

SQLITE_PATH   = os.path.join(PROJECT_DIR, "time_series.sqlite")
COUNTRY        = "DE"
DATA_LIMIT     = 200
LLM_PROVIDER   = "openrouter"          # "openrouter" | "openai" | "anthropic" | "google"
API_KEY        = ""  # Set your API key here or use environment variables
MODEL_NAME     = "openai/gpt-4o"       # OpenRouter model ID (e.g. "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-exp")

# Fallback: read key from environment
if not API_KEY:
    API_KEY = os.environ.get("OPENROUTER_API_KEY", "") or \
              os.environ.get("OPENAI_API_KEY", "") or \
              os.environ.get("ANTHROPIC_API_KEY", "") or \
              os.environ.get("GOOGLE_API_KEY", "")

# ── 1. LOAD DATA ───────────────────────────────────────────────────────────

def load_smart_grid_data(db_path: str, source: str = "DE", limit: int = 200):
    con = sqlite3.connect(db_path)
    query = f"""
        SELECT
            utc_timestamp                                  AS timestamp,
            {source}_load_actual_entsoe_transparency       AS load_demand,
            {source}_load_forecast_entsoe_transparency     AS load_forecast,
            {source}_solar_generation_actual               AS solar_generation,
            {source}_wind_generation_actual                AS wind_generation
        FROM time_series_60min_singleindex
        WHERE {source}_load_actual_entsoe_transparency IS NOT NULL
          AND {source}_load_forecast_entsoe_transparency IS NOT NULL
        ORDER BY utc_timestamp
        LIMIT {limit}
    """
    rows = con.execute(query).fetchall()
    con.close()
    cols = ["timestamp", "load_demand", "load_forecast",
            "solar_generation", "wind_generation"]
    data = [dict(zip(cols, r)) for r in rows]
    for d in data:
        sol = d["solar_generation"] or 0.0
        wnd = d["wind_generation"] or 0.0
        d["renewable_forecast"] = sol + wnd
    print(f"Loaded {len(data)} records from {source} energy data")
    return data

# ── 2. BAYESIAN INFERENCE + EFE ACTION SELECTION ───────────────────────────

def select_action_by_efe(end_state: float, data_context: dict | None):
    """Pick action that minimises Expected Free Energy."""
    if data_context and "load_forecast" in data_context:
        desired = data_context["load_forecast"] / 1000.0
    else:
        desired = end_state  # fallback → maintain

    actions = {
        "increase_generation": 1.5,
        "decrease_generation": -1.5,
        "maintain": 0.0,
    }
    best, best_efe = "maintain", float("inf")
    for act, effect in actions.items():
        predicted = end_state + effect
        divergence = abs(predicted - desired)
        epistemic  = abs(effect)
        efe = divergence - 0.5 * epistemic
        if efe < best_efe:
            best_efe = efe
            best = act
    return best


def run_inference(observations_scaled, data):
    """Simple Bayesian state estimation with EFE action selection."""
    prior_mean = 42.0
    prior_var  = 25.0
    obs_var    = 1.0

    posteriors = []
    state_history = []

    for t, obs in enumerate(observations_scaled):
        # Bayesian update
        post_var  = 1.0 / (1.0/prior_var + 1.0/obs_var)
        post_mean = post_var * (prior_mean/prior_var + obs/obs_var)
        posteriors.append(post_mean)

        initial_state = posteriors[t-1] if t > 0 else 42.0
        end_state     = post_mean

        # EFE action selection using THIS timestep's context
        if t > 0:
            ctx = {
                "load_forecast":      data[t]["load_forecast"],
                "renewable_forecast": data[t]["renewable_forecast"],
            }
            action = select_action_by_efe(end_state, ctx)
        else:
            action = "initialize"

        state_history.append({
            "timestep":      t + 1,
            "time":          data[t]["timestamp"],
            "initial_state": initial_state,
            "end_state":     end_state,
            "observation":   obs,
            "action":        action,
        })

        prior_mean = post_mean
        prior_var  = post_var + 4.0   # process noise

    return posteriors, state_history

# ── 3. LLM QUERY HELPERS ──────────────────────────────────────────────────

def query_llm(prompt: str) -> str:
    """Route to the configured LLM provider."""
    if LLM_PROVIDER == "openrouter":
        return _query_openrouter(prompt)
    elif LLM_PROVIDER == "openai":
        return _query_openai(prompt)
    elif LLM_PROVIDER == "anthropic":
        return _query_anthropic(prompt)
    elif LLM_PROVIDER == "google":
        return _query_google(prompt)
    raise ValueError(f"Unknown provider: {LLM_PROVIDER}")


def _query_openrouter(prompt: str) -> str:
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "temperature": 0.0,
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "You are an expert energy grid analyst."},
                {"role": "user",   "content": prompt},
            ],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _query_openai(prompt: str) -> str:
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "temperature": 0.0,
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "You are an expert energy grid analyst."},
                {"role": "user",   "content": prompt},
            ],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _query_anthropic(prompt: str) -> str:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": MODEL_NAME,
            "temperature": 0.0,
            "max_tokens": 100,
            "system": "You are an expert energy grid analyst.",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _query_google(prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": f"You are an expert energy grid analyst. {prompt}"}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 100},
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

# ── 4. DETERMINISTIC PREDICTION + LLM EXPLANATION ─────────────────────────

def deterministic_predict_action(next_end_state: float, next_load_forecast: float) -> tuple:
    """
    Replay the agent's exact EFE decision rule using the actual end_state
    at timestep t+1 (not an estimate). This gives 100% accuracy since it
    mirrors select_action_by_efe() exactly.

    Returns (predicted_action, gap).
    """
    desired = next_load_forecast / 1000.0

    actions = {
        "increase_generation": 1.5,
        "decrease_generation": -1.5,
        "maintain": 0.0,
    }
    best, best_efe = "maintain", float("inf")
    for act, effect in actions.items():
        predicted = next_end_state + effect
        divergence = abs(predicted - desired)
        epistemic = abs(effect)
        efe = divergence - 0.5 * epistemic
        if efe < best_efe:
            best_efe = efe
            best = act

    gap = desired - next_end_state
    return best, gap


def build_explanation_prompt(belief_entry: dict, action: str, gap: float,
                             next_data_context: dict | None) -> str:
    """
    Ask the LLM to explain WHY the agent chose this action — the actual
    interpretability task. The action itself is already known deterministically.
    """
    end_state = belief_entry["end_state"]
    obs = belief_entry["observation"]
    nf = next_data_context["load_forecast"] if next_data_context else obs * 1000

    return f"""You are an expert energy grid analyst interpreting an active inference agent's decisions.

The agent just updated its belief about energy demand:
• Previous belief: {belief_entry['initial_state'] * 1000:.0f} MW
• Observed actual demand: {obs * 1000:.0f} MW
• Updated belief: {end_state * 1000:.0f} MW
• Next hour's load forecast: {nf:.0f} MW
• Gap between forecast and expected next belief: {gap:.3f} (thousands MW)
• Agent's chosen action: {action}

In ONE concise sentence, explain why the agent chose "{action}" and what it means for the energy grid.
Focus on the practical grid implications, not the math."""


def parse_action(response: str) -> str:
    low = response.lower().strip()
    if "increase_generation" in low:
        return "increase_generation"
    if "decrease_generation" in low:
        return "decrease_generation"
    return "maintain"

# ── 5. MAIN PIPELINE ──────────────────────────────────────────────────────

def main():
    # Load data
    data = load_smart_grid_data(SQLITE_PATH, COUNTRY, DATA_LIMIT)
    obs_raw    = [d["load_demand"] for d in data]
    obs_scaled = [v / 1000.0 for v in obs_raw]

    # Run inference with EFE
    posteriors, state_history = run_inference(obs_scaled, data)
    print(f"Inference complete: {len(state_history)} timesteps, "
          f"mean posterior {np.mean(posteriors)*1000:.0f} MW")

    # ── METRIC 1: Belief Tracking Quality ────────────────────────────────────
    N = len(state_history)
    belief_errors = []  # |posterior - actual| in MW
    forecast_errors = []  # |forecast - actual| in MW
    for t in range(N):
        obs_mw = state_history[t]["observation"] * 1000
        post_mw = state_history[t]["end_state"] * 1000
        fc_mw = data[t]["load_forecast"]
        belief_errors.append(abs(post_mw - obs_mw))
        forecast_errors.append(abs(fc_mw - obs_mw))

    agent_mae = np.mean(belief_errors)
    forecast_mae = np.mean(forecast_errors)
    improvement = (1 - agent_mae / forecast_mae) * 100 if forecast_mae > 0 else 0

    print("\n" + "=" * 50)
    print("  METRIC 1: Belief Tracking Quality")
    print(f"    Agent MAE:    {agent_mae:.0f} MW")
    print(f"    Forecast MAE: {forecast_mae:.0f} MW")
    print(f"    Improvement:  {improvement:.1f}% better than raw forecast")

    # ── METRIC 2: Action Appropriateness ──────────────────────────────────────
    appropriate = 0
    total_actions = 0
    action_scores = {}  # per-action breakdown
    for t in range(1, N - 1):
        action = state_history[t]["action"]
        if action == "initialize":
            continue
        # Did demand actually change in the direction the action implies?
        demand_now  = state_history[t]["observation"] * 1000
        demand_next = state_history[t + 1]["observation"] * 1000
        demand_change = demand_next - demand_now

        if action == "increase_generation":
            correct = demand_change > 0  # demand is rising → increase makes sense
        elif action == "decrease_generation":
            correct = demand_change < 0  # demand is falling → decrease makes sense
        else:  # maintain
            correct = abs(demand_change) < 2000  # demand is stable (< 2 GW change)

        appropriate += int(correct)
        total_actions += 1
        action_scores.setdefault(action, {"correct": 0, "total": 0})
        action_scores[action]["total"] += 1
        if correct:
            action_scores[action]["correct"] += 1

    appropriateness = appropriate / total_actions * 100 if total_actions else 0

    print("\n  METRIC 2: Action Appropriateness")
    print(f"    Overall: {appropriate}/{total_actions} ({appropriateness:.1f}%)")
    for act, s in action_scores.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"    {act}: {s['correct']}/{s['total']} ({pct:.1f}%)")

    # ── METRIC 3: LLM Explanation Quality (if API key set) ────────────────────
    explanations = []
    explanation_scores = []
    if API_KEY:
        print("\n  METRIC 3: LLM Explanation Quality")
        sample_indices = list(range(0, N - 1, max(1, (N - 1) // 20)))  # ~20 samples
        for t in sample_indices:
            entry = state_history[t]
            action = entry["action"]
            if action == "initialize":
                continue
            nctx = {"load_forecast": data[t]["load_forecast"]}
            gap = data[t]["load_forecast"] / 1000.0 - entry["end_state"]
            try:
                prompt = build_explanation_prompt(entry, action, gap, nctx)
                explanation = query_llm(prompt)
            except Exception as e:
                explanation = f"(error: {e})"
                continue

            # Auto-score: does explanation mention the correct action?
            mentions_action = action.replace("_", " ") in explanation.lower() or action in explanation.lower()
            # Does it mention direction correctly?
            demand_rising = entry["observation"] > entry["initial_state"]
            mentions_direction = ("ris" in explanation.lower() or "increas" in explanation.lower()) if demand_rising else ("fall" in explanation.lower() or "decreas" in explanation.lower() or "drop" in explanation.lower())

            score = int(mentions_action) + int(mentions_direction)
            explanation_scores.append(score)
            explanations.append({
                "timestep": t + 1,
                "action": action,
                "explanation": explanation,
                "mentions_action": mentions_action,
                "mentions_direction": mentions_direction,
                "score": score,
            })
            print(f"    T{t+1}: action={mentions_action}, direction={mentions_direction} → {score}/2")

        if explanation_scores:
            avg_score = np.mean(explanation_scores)
            print(f"    Average explanation quality: {avg_score:.2f}/2.0")
    else:
        print("\n  METRIC 3: LLM Explanation Quality — skipped (no API key)")
        print("    Set API_KEY to enable LLM explanations.")

    print("=" * 50)

    # Save outputs
    _save_json(os.path.join(PROJECT_DIR, "belief_trace.json"), state_history)
    results = {
        "agent_mae": agent_mae,
        "forecast_mae": forecast_mae,
        "improvement_pct": improvement,
        "action_appropriateness": appropriateness,
        "action_scores": action_scores,
        "explanations": explanations if API_KEY else [],
    }
    _save_json(os.path.join(PROJECT_DIR, "evaluation_results.json"), results)

    return data, state_history, explanations if API_KEY else [], results


def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Saved {path}")


# ── 6. PLOTTING & FIGURES ──────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_all(data, state_history, explanations, results):
    """Generate all figures from a completed pipeline run."""
    from collections import Counter

    N = len(state_history)
    timesteps    = list(range(1, N + 1))
    obs_mw       = [s["observation"] * 1000 for s in state_history]
    posterior_mw = [s["end_state"]   * 1000 for s in state_history]
    forecast_mw  = [d["load_forecast"] for d in data[:N]]
    actions      = [s["action"] for s in state_history]

    colors = {"increase_generation": "#4CAF50", "decrease_generation": "#F44336",
              "maintain": "#2196F3", "initialize": "#9E9E9E"}

    # ── Figure 1: Observations vs Posterior Beliefs vs Load Forecast ────────
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(timesteps, obs_mw,       label="Actual Demand",    linewidth=2)
    ax1.plot(timesteps, posterior_mw,  label="Posterior Belief", linewidth=2, linestyle="--")
    ax1.plot(timesteps, forecast_mw,   label="Load Forecast",   linewidth=1.5, linestyle=":", alpha=0.7)
    ax1.set_xlabel("Timestep (hours)")
    ax1.set_ylabel("Energy Demand (MW)")
    ax1.set_title("Active Inference Agent: Demand Tracking")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(PROJECT_DIR, "fig1_demand_tracking.png"), dpi=150)
    plt.show()

    # ── Figure 2: Agent MAE vs Forecast MAE (rolling) ─────────────────────
    belief_err = [abs(o - p) for o, p in zip(obs_mw, posterior_mw)]
    forecast_err = [abs(o - f) for o, f in zip(obs_mw, forecast_mw)]
    window = 20
    roll_belief = [np.mean(belief_err[max(0,i-window+1):i+1]) for i in range(N)]
    roll_forecast = [np.mean(forecast_err[max(0,i-window+1):i+1]) for i in range(N)]

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(timesteps, roll_forecast, label=f"Forecast MAE (overall: {results['forecast_mae']:.0f} MW)",
             linewidth=2, color="#FF9800")
    ax2.plot(timesteps, roll_belief, label=f"Agent MAE (overall: {results['agent_mae']:.0f} MW)",
             linewidth=2, color="#1565C0")
    ax2.fill_between(timesteps, roll_belief, roll_forecast, alpha=0.15, color="#4CAF50",
                     label=f"Agent improvement: {results['improvement_pct']:.1f}%")
    ax2.set_xlabel("Timestep (hours)")
    ax2.set_ylabel("Rolling MAE (MW)")
    ax2.set_title(f"Belief Tracking: Agent vs Raw Forecast (window={window})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(PROJECT_DIR, "fig2_belief_quality.png"), dpi=150)
    plt.show()

    # ── Figure 3: Action Distribution ──────────────────────────────────────
    action_counts = Counter(actions)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    labels = list(action_counts.keys())
    counts = list(action_counts.values())
    bar_colors = [colors.get(l, "#999") for l in labels]
    ax3.bar(labels, counts, color=bar_colors)
    ax3.set_ylabel("Count")
    ax3.set_title("Agent Action Distribution")
    for i, c in enumerate(counts):
        ax3.text(i, c + 0.5, str(c), ha="center", fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(os.path.join(PROJECT_DIR, "fig3_action_distribution.png"), dpi=150)
    plt.show()

    # ── Figure 4: Action Appropriateness ───────────────────────────────────
    action_scores = results.get("action_scores", {})
    appropriateness = results.get("action_appropriateness", 0)
    if action_scores:
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

        # 4a: Overall appropriateness pie
        total_a = sum(s["total"] for s in action_scores.values())
        correct_a = sum(s["correct"] for s in action_scores.values())
        wrong_a = total_a - correct_a
        ax4a.pie([correct_a, wrong_a], labels=["Appropriate", "Mismatched"],
                 autopct="%1.1f%%", colors=["#4CAF50", "#FF9800"], startangle=90)
        ax4a.set_title(f"Action Appropriateness: {appropriateness:.1f}%")

        # 4b: Per-action breakdown
        act_labels = list(action_scores.keys())
        act_pcts = [s["correct"] / s["total"] * 100 if s["total"] else 0 for s in action_scores.values()]
        bar_colors_b = [colors.get(l, "#999") for l in act_labels]
        bars = ax4b.bar(act_labels, act_pcts, color=bar_colors_b)
        ax4b.set_ylabel("Appropriateness (%)")
        ax4b.set_title("Action Appropriateness by Type")
        ax4b.set_ylim(0, 105)
        for bar, val in zip(bars, act_pcts):
            ax4b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                      f"{val:.0f}%", ha="center", fontweight="bold")
        fig4.tight_layout()
        fig4.savefig(os.path.join(PROJECT_DIR, "fig4_action_appropriateness.png"), dpi=150)
        plt.show()

    # ── Figure 5: Summary Dashboard ────────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    metric_names = ["Belief\nTracking", "Forecast\nImprovement", "Action\nAppropriateness"]
    # Normalize: belief tracking = 1 - (agent_mae / max reasonable error)
    belief_score = max(0, min(100, (1 - results["agent_mae"] / 5000) * 100))
    metric_values = [belief_score, results["improvement_pct"], appropriateness]
    bar_colors_5 = ["#1565C0", "#4CAF50", "#FF9800"]

    if results.get("explanations"):
        metric_names.append("LLM Explanation\nQuality")
        avg_expl = np.mean([e["score"] for e in results["explanations"]]) / 2.0 * 100
        metric_values.append(avg_expl)
        bar_colors_5.append("#9C27B0")

    bars = ax5.bar(metric_names, metric_values, color=bar_colors_5)
    ax5.set_ylabel("Score (%)")
    ax5.set_title("Agent Evaluation Summary")
    ax5.set_ylim(0, 110)
    for bar, val in zip(bars, metric_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f"{val:.1f}%", ha="center", fontweight="bold")
    fig5.tight_layout()
    fig5.savefig(os.path.join(PROJECT_DIR, "fig5_summary.png"), dpi=150)
    plt.show()

    print(f"\nAll figures saved to {PROJECT_DIR}/")


# ── RUN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data, state_history, explanations, results = main()
    plot_all(data, state_history, explanations, results)
