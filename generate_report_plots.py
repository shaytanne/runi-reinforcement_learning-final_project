from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION — EDIT THIS PART
# ============================================================

# Top-level directory that contains all experiment result folders
BASE_DIR = Path("official_results")

# Where generated plot PNGs will be saved
OUTPUT_DIR = Path("report_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact experiment folders
RUN_DIRS = {
    "SG_DQN": BASE_DIR / "SET5_DQN_SG_20260303-192956",
    "SG_PPO": BASE_DIR / "SET5_PPO_SG_20260303-202219",
    "SG_A2C": BASE_DIR / "SET5_A2C_SG_20260303-195355",
    "7A": BASE_DIR / "EXP7A_PPO_GAE_ONLY_20260307-115948",
    "7A_EXTEND": BASE_DIR / "EXP7A_PPO_GAE_EXTEND_20260307-141506",
    "7A_FINETUNE": BASE_DIR / "EXP7A_PPO_GAE_FINE_TUNE_20260307-144459",
    "7C": BASE_DIR / "EXP7C_PPO_GAE_RGB_20260307-193916",
    # Optional later:
    # "4B": BASE_DIR / "EXP4B_DDQN_PER_KDB_20260308-....",
}

# Offsets for resumed runs, so they appear after episode 3000 on training plots
RUN_OFFSETS = {
    "SG_DQN": 0,
    "SG_PPO": 0,
    "SG_A2C": 0,
    "7A": 0,
    "7A_EXTEND": 3000,
    "7A_FINETUNE": 3000,
    "7C": 0,
    # "4B": 0,
}

WINDOW_SG = 50
WINDOW_KDB = 100

REQUIRED_TRAIN_COLS = {"episode", "reward", "steps", "success"}
REQUIRED_INFER_COLS = {"episode", "reward", "steps", "success"}


# ============================================================
# FILE LOOKUP
# ============================================================

def run_file(run_name: str, suffix: str) -> Path:
    """
    Finds exactly one CSV file inside the given run folder whose filename ends with `suffix`.
    Example:
        run_file("7A", "training_log.csv")
        -> resolves e.g. to .../7A_training_log.csv
    """
    root = RUN_DIRS[run_name]

    if not root.exists():
        raise FileNotFoundError(f"Experiment folder does not exist: {root}")

    matches = [p for p in root.rglob("*.csv") if p.name.endswith(suffix)]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple files ending with '{suffix}' found under {root}:\n" +
            "\n".join(str(m) for m in matches)
        )

    raise FileNotFoundError(f"No file ending with '{suffix}' found under {root}")


# ============================================================
# CSV / DATA HELPERS
# ============================================================

def load_csv(path: Path, required_cols: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def summarize_inference(df: pd.DataFrame) -> dict[str, float]:
    return {
        "success": float(df["success"].mean()),
        "steps": float(df["steps"].mean()),
        "reward": float(df["reward"].mean()),
        "n_episodes": float(len(df)),
    }


def collect_training(run_specs: dict[str, tuple[str, str]]) -> list[tuple[str, pd.DataFrame, int]]:
    """
    run_specs: dict[label] = (run_name, suffix)
    """
    collected = []
    for label, (run_name, suffix) in run_specs.items():
        try:
            path = run_file(run_name, suffix)
            df = load_csv(path, REQUIRED_TRAIN_COLS)
            offset = int(RUN_OFFSETS.get(run_name, 0))
            collected.append((label, df, offset))
            print(f"[OK] Training file for {label}: {path}")
        except Exception as e:
            print(f"[WARN] Skipping training run '{label}': {e}")
    return collected


def collect_inference(run_specs: dict[str, tuple[str, str]]) -> list[tuple[str, pd.DataFrame]]:
    """
    run_specs: dict[label] = (run_name, suffix)
    """
    collected = []
    for label, (run_name, suffix) in run_specs.items():
        try:
            path = run_file(run_name, suffix)
            df = load_csv(path, REQUIRED_INFER_COLS)
            collected.append((label, df))
            print(f"[OK] Inference file for {label}: {path}")
        except Exception as e:
            print(f"[WARN] Skipping inference run '{label}': {e}")
    return collected


# ============================================================
# PLOTTING HELPERS
# ============================================================

def save_training_overlay(
    run_specs: dict[str, tuple[str, str]],
    metric: str,
    window: int,
    title: str,
    ylabel: str,
    output_name: str,
) -> None:
    items = collect_training(run_specs)
    if not items:
        print(f"[WARN] No training runs available for {output_name}")
        return

    plt.figure(figsize=(8, 5))
    for label, df, offset in items:
        x = df["episode"] + offset
        y = rolling_mean(df[metric], window)
        plt.plot(x, y, label=label)

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / output_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")


def save_inference_dual_axis_bars(
    run_specs: dict[str, tuple[str, str]],
    title: str,
    output_name: str,
    success_label: str = "Success rate",
    steps_label: str = "Average steps",
    success_color: str = "tab:blue",
    steps_color: str = "tab:orange",
) -> None:
    """
    Creates one combined inference figure:
    - one x-label per experiment
    - two bars per experiment
    - left y-axis = success
    - right y-axis = steps
    """
    items = collect_inference(run_specs)
    if not items:
        print(f"[WARN] No inference runs available for {output_name}")
        return

    labels = []
    success_vals = []
    step_vals = []

    for label, df in items:
        stats = summarize_inference(df)
        labels.append(label)
        success_vals.append(stats["success"])
        step_vals.append(stats["steps"])

    x = np.arange(len(labels))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    bars_success = ax1.bar(
        x - width / 2,
        success_vals,
        width,
        label="Success",
        color=success_color,
        alpha=0.9,
    )
    bars_steps = ax2.bar(
        x + width / 2,
        step_vals,
        width,
        label="Avg steps",
        color=steps_color,
        alpha=0.9,
    )

    ax1.set_xlabel("Experiment")
    ax1.set_ylabel(success_label)
    ax2.set_ylabel(steps_label)
    ax1.set_title(title)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")

    ax1.set_ylim(0, max(1.05, max(success_vals) * 1.1))

    handles = [bars_success, bars_steps]
    legend_labels = ["Success", "Avg steps"]
    ax1.legend(handles, legend_labels, loc="upper center", ncol=2)

    plt.tight_layout()

    out_path = OUTPUT_DIR / output_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")


def save_greedy_stochastic_dual_axis_bars(
    experiment_specs: dict[str, dict[str, tuple[str, str]]],
    title: str,
    output_name: str,
    success_label: str = "Inference success rate",
    steps_label: str = "Average inference steps",
    greedy_color: str = "tab:red",
    stochastic_color: str = "tab:green",
) -> None:
    """
    Creates one combined figure for greedy vs stochastic:
    - single x-label per experiment
    - four bars per experiment:
        greedy success, stochastic success (left axis)
        greedy steps, stochastic steps (right axis)
    - colors distinguish greedy vs stochastic
    """
    labels = []
    greedy_success = []
    stochastic_success = []
    greedy_steps = []
    stochastic_steps = []

    for experiment_label, mode_specs in experiment_specs.items():
        try:
            greedy_path = run_file(*mode_specs["greedy"])
            stochastic_path = run_file(*mode_specs["stochastic"])

            greedy_df = load_csv(greedy_path, REQUIRED_INFER_COLS)
            stochastic_df = load_csv(stochastic_path, REQUIRED_INFER_COLS)

            greedy_stats = summarize_inference(greedy_df)
            stochastic_stats = summarize_inference(stochastic_df)

            labels.append(experiment_label)
            greedy_success.append(greedy_stats["success"])
            stochastic_success.append(stochastic_stats["success"])
            greedy_steps.append(greedy_stats["steps"])
            stochastic_steps.append(stochastic_stats["steps"])

            print(f"[OK] Greedy file for {experiment_label}: {greedy_path}")
            print(f"[OK] Stochastic file for {experiment_label}: {stochastic_path}")

        except Exception as e:
            print(f"[WARN] Skipping experiment '{experiment_label}' in greedy/stochastic plot: {e}")

    if not labels:
        print(f"[WARN] No experiments available for {output_name}")
        return

    x = np.arange(len(labels))
    width = 0.18

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Left axis: success
    bars_greedy_success = ax1.bar(
        x - 1.5 * width,
        greedy_success,
        width,
        label="Greedy success",
        color=greedy_color,
        alpha=0.9,
    )
    bars_stochastic_success = ax1.bar(
        x - 0.5 * width,
        stochastic_success,
        width,
        label="Stochastic success",
        color=stochastic_color,
        alpha=0.9,
    )

    # Right axis: steps
    bars_greedy_steps = ax2.bar(
        x + 0.5 * width,
        greedy_steps,
        width,
        label="Greedy steps",
        color=greedy_color,
        alpha=0.45,
    )
    bars_stochastic_steps = ax2.bar(
        x + 1.5 * width,
        stochastic_steps,
        width,
        label="Stochastic steps",
        color=stochastic_color,
        alpha=0.45,
    )

    ax1.set_xlabel("Experiment")
    ax1.set_ylabel(success_label)
    ax2.set_ylabel(steps_label)
    ax1.set_title(title)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(0, max(1.05, max(stochastic_success + greedy_success) * 1.1))

    handles = [
        bars_greedy_success,
        bars_stochastic_success,
        bars_greedy_steps,
        bars_stochastic_steps,
    ]
    legend_labels = [
        "Greedy success",
        "Stochastic success",
        "Greedy steps",
        "Stochastic steps",
    ]
    ax1.legend(handles, legend_labels, loc="upper center", ncol=2)

    plt.tight_layout()

    out_path = OUTPUT_DIR / output_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")


def print_numeric_summaries(run_specs: dict[str, tuple[str, str]], title: str) -> None:
    print(f"\n=== {title} ===")
    items = collect_inference(run_specs)
    for label, df in items:
        stats = summarize_inference(df)
        print(
            f"{label:>22} | "
            f"success={stats['success']:.4f} | "
            f"steps={stats['steps']:.2f} | "
            f"reward={stats['reward']:.2f} | "
            f"n={int(stats['n_episodes'])}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print(f"Using BASE_DIR={BASE_DIR.resolve()}")
    print(f"Saving plots to {OUTPUT_DIR.resolve()}\n")

    # Figure 1: SimpleGrid comparison
    sg_training_runs = {
        "DQN": ("SG_DQN", "training_log.csv"),
        "PPO": ("SG_PPO", "training_log.csv"),
        "A2C": ("SG_A2C", "training_log.csv"),
    }

    sg_inference_runs = {
        "DQN": ("SG_DQN", "inference_log.csv"),
        "PPO": ("SG_PPO", "inference_log.csv"),
        "A2C": ("SG_A2C", "inference_log.csv"),
    }

    save_training_overlay(
        run_specs=sg_training_runs,
        metric="success",
        window=WINDOW_SG,
        title="SimpleGrid training success",
        ylabel="Rolling success rate",
        output_name="figure_1a_sg_training_success.png",
    )

    save_inference_dual_axis_bars(
        run_specs=sg_inference_runs,
        title="SimpleGrid inference comparison",
        output_name="figure_1b_sg_inference_success_steps_combined.png",
        success_label="Inference success rate",
        steps_label="Average inference steps",
        success_color="tab:blue",
        steps_color="tab:orange",
    )

    # Figure 2: KDB PPO progression
    kdb_ppo_training_runs = {
        "7A grayscale": ("7A", "training_log.csv"),
        "7A extend": ("7A_EXTEND", "training_log.csv"),
        "7A fine-tune": ("7A_FINETUNE", "training_log.csv"),
        "7C RGB": ("7C", "training_log.csv"),
    }

    save_training_overlay(
        run_specs=kdb_ppo_training_runs,
        metric="success",
        window=WINDOW_KDB,
        title="KDB PPO training success",
        ylabel="Rolling success rate",
        output_name="figure_2a_kdb_ppo_training_success.png",
    )

    save_training_overlay(
        run_specs=kdb_ppo_training_runs,
        metric="reward",
        window=WINDOW_KDB,
        title="KDB PPO training reward",
        ylabel="Rolling mean reward",
        output_name="figure_2b_kdb_ppo_training_reward.png",
    )

    # Figure 3: 7A vs 7C grayscale vs RGB (stochastic)
    rgb_vs_gray_runs = {
        "7A grayscale": ("7A", "inference_log_stochastic.csv"),
        "7C RGB": ("7C", "inference_log_stochastic.csv"),
    }

    save_inference_dual_axis_bars(
        run_specs=rgb_vs_gray_runs,
        title="KDB PPO: grayscale vs RGB (stochastic inference)",
        output_name="figure_3_kdb_gray_vs_rgb_combined.png",
        success_label="Stochastic inference success rate",
        steps_label="Average stochastic inference steps",
        success_color="tab:blue",
        steps_color="tab:orange",
    )

    # Figure 4: Greedy vs stochastic PPO
    greedy_stochastic_experiments = {
        "7A": {
            "greedy": ("7A", "inference_log_greedy.csv"),
            "stochastic": ("7A", "inference_log_stochastic.csv"),
        },
        "7A extend": {
            "greedy": ("7A_EXTEND", "inference_log_greedy.csv"),
            "stochastic": ("7A_EXTEND", "inference_log_stochastic.csv"),
        },
        "7C": {
            "greedy": ("7C", "inference_log_greedy.csv"),
            "stochastic": ("7C", "inference_log_stochastic.csv"),
        },
    }

    save_greedy_stochastic_dual_axis_bars(
        experiment_specs=greedy_stochastic_experiments,
        title="KDB PPO: greedy vs stochastic inference",
        output_name="figure_4_kdb_greedy_vs_stochastic_combined.png",
        success_label="Inference success rate",
        steps_label="Average inference steps",
        greedy_color="tab:red",
        stochastic_color="tab:green",
    )

    # Optional: print summaries to terminal
    print_numeric_summaries(sg_inference_runs, "SimpleGrid inference summaries")
    print_numeric_summaries(rgb_vs_gray_runs, "KDB grayscale vs RGB summaries")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())