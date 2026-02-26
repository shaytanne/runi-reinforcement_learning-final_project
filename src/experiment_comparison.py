import json
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Dict

from src.utils import save_fig 



def _load_experiment_runs(run_directories: Dict[str, str]) -> list[dict]:
    """
    Loads and validates data for a set of experiments.
    Returns a list of experiment dicts, each containing all available data for that run.
    All comparison plotting functions consume this output — call this once, pass result to any plot.

    :param run_directories: dict like {'PPO_Shaped': 'results/CB3_PPO_.../', 'DQN_Baseline': 'results/CB1_.../'}
    :returns: list of dicts, one per experiment, with keys:
        - label         (str)               experiment name
        - log_dir       (str)               path to results folder
        - training_df   (pd.DataFrame|None) training_log.csv
        - milestone_df  (pd.DataFrame|None) milestone_log.csv (None for SimpleGrid)
        - inference_df  (pd.DataFrame|None) inference_log.csv
        - report        (dict|None)         experiment_report.json
    """
    runs = []
    for label, log_dir in run_directories.items():
        if not os.path.isdir(log_dir):
            print(f"WARNING: directory not found, skipping '{label}': {log_dir}")
            continue

        def _load_csv(filename):
            path = os.path.join(log_dir, filename)
            return pd.read_csv(path) if os.path.exists(path) else None

        def _load_json(filename):
            path = os.path.join(log_dir, filename)
            if not os.path.exists(path):
                return None
            with open(path) as f:
                return json.load(f)

        runs.append({
            "label":        label,
            "log_dir":      log_dir,
            "training_df":  _load_csv("training_log.csv"),
            "action_df":    _load_csv("action_dist_training.csv"), 
            "milestone_df": _load_csv("milestone_log.csv"),
            "inference_df": _load_csv("inference_log.csv"),
            "report":       _load_json("experiment_report.json"),
        })

    if not runs:
        print("WARNING: no valid experiment directories found.")
    return runs


def plot_reward_comparison(runs: list[dict], window: int = 50, save_dir: str = "results") -> None:
    """Reward curves across multiple experiments"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for run in runs:
        df = run["training_df"]
        if df is None:
            continue
        reward = df["reward"]
        if len(df) > window:
            reward = reward.rolling(window).mean()
        ax.plot(df["episode"], reward, linewidth=2, label=run["label"])

    ax.set_title(f"Reward Comparison (Rolling {window} eps)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, save_dir, "reward_comparison.png")


def plot_success_rate_comparison(runs: list[dict], window: int = 50, save_dir: str = "results") -> None:
    """Success rate curves across multiple experiments"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for run in runs:
        df = run["training_df"]
        if df is None:
            continue
        success = pd.to_numeric(df["success"], errors="coerce").fillna(0)
        if len(df) > window:
            success = success.rolling(window).mean()
        ax.plot(df["episode"], success, linewidth=2, label=run["label"])

    ax.set_title(f"Success Rate Comparison (Rolling {window} eps)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, save_dir, "success_rate_comparison.png")


def plot_milestone_comparison(runs: list[dict], milestone: str = "got_key", save_dir: str = "results") -> None:
    """Cumulative milestone count across experiments (KDB only)."""
    milestone_labels = {
        "got_key": "Key Pickup", "opened_door": "Door Opened",
        "has_crossed_door": "Room Crossed", "got_ball": "Ball Pickup",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for run in runs:
        df = run["milestone_df"]
        if df is None or milestone not in df.columns:
            continue
        cumsum = pd.to_numeric(df[milestone], errors="coerce").fillna(0).cumsum()
        ax.plot(df["episode"], cumsum, linewidth=2, label=run["label"])

    ax.set_title(f"Cumulative {milestone_labels.get(milestone, milestone)} — Cross-Experiment")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, save_dir, f"milestone_comparison_{milestone}.png")


def plot_summary_bar(runs: list[dict], save_dir: str = "results") -> None:
    """Bar chart of inference success rate and avg steps across experiments."""
    labels, success_rates, avg_steps = [], [], []
    for run in runs:
        report = run["report"]
        if report is None:
            continue
        results = report.get("results", {})
        labels.append(run["label"])
        success_rates.append(results.get("inference_success_rate", 0) * 100)
        avg_steps.append(results.get("inference_avg_steps", 0))

    if not labels:
        print("No experiment reports found.")
        return

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(x, success_rates, color="steelblue", alpha=0.8)
    ax1.bar_label(bars1, fmt="%.1f%%", padding=3)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Success Rate (%)"); ax1.set_title("Inference Success Rate")
    ax1.set_ylim(0, 110); ax1.grid(True, axis="y", alpha=0.3)

    bars2 = ax2.bar(x, avg_steps, color="coral", alpha=0.8)
    ax2.bar_label(bars2, fmt="%.0f", padding=3)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Avg Steps"); ax2.set_title("Inference Avg Steps")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_dir, "summary_bar.png")


def generate_summary_table(runs: list[dict], save_dir: str = "results") -> None:
    """CSV summary table — one row per experiment."""
    rows = []
    for run in runs:
        report = run["report"]
        if report is None:
            continue
        rows.append({
            "experiment":    run["label"],
            "algo":          report.get("meta", {}).get("algo", ""),
            "env":           report.get("meta", {}).get("env", ""),
            "train_episodes": report.get("results", {}).get("train_episodes", ""),
            "success_rate_%": round(report.get("results", {}).get("inference_success_rate", 0) * 100, 1),
            "avg_reward":    round(report.get("results", {}).get("inference_avg_reward", 0), 3),
            "avg_steps":     round(report.get("results", {}).get("inference_avg_steps", 0), 1),
            "final_epsilon": report.get("results", {}).get("train_final_epsilon", "n/a"),
            "train_time":    report.get("performance", {}).get("train", ""),
        })
    if not rows:
        print("No experiment reports found.")
        return
    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "summary_table.csv")
    df.to_csv(save_path, index=False)
    print(f"Summary table saved to: {save_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    runs = _load_experiment_runs({
        "exp name": "results/exp name",
    })

    plot_reward_comparison(runs=runs, save_dir="results/comparison")
    plot_success_rate_comparison(runs=runs, save_dir="results/comparison")
    plot_milestone_comparison(runs=runs, milestone="got_key", save_dir="results/comparison")
    plot_milestone_comparison(runs=runs, milestone="opened_door", save_dir="results/comparison")
    plot_summary_bar(runs=runs, save_dir="results/comparison")
    generate_summary_table(runs=runs, save_dir="results/comparison")