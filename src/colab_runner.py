"""
Colab Launcher — run this notebook-style in Google Colab.

Usage:
  1. Open Google Colab
  2. Set runtime to GPU (Runtime > Change runtime type > T4 GPU)
  3. Copy-paste this into a cell (or upload to repo and !python colab_runner.py)
"""

# === Cell 1: Setup ===
import subprocess, os

# Clone repo (skip if already cloned)
REPO = "shaytanne/runi-reinforcement_learning-final_project"
BRANCH = "infra_merge"

if not os.path.exists("runi-reinforcement_learning-final_project"):
    subprocess.run(["git", "clone", "-b", BRANCH, f"https://github.com/{REPO}.git"], check=True)

os.chdir("runi-reinforcement_learning-final_project")

# Install deps (Colab already has torch, numpy, matplotlib, pandas)
subprocess.run(["pip", "install", "-q", "minigrid", "gymnasium", "imageio[ffmpeg]", 
                 "pygame", "pyvirtualdisplay", "opencv-python"], check=True)

# Virtual display for headless rendering (Colab has no screen)
subprocess.run(["apt-get", "-qq", "install", "-y", "xvfb"], check=True)
from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 400))
display.start()

# === Cell 2: Verify GPU ===
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# === Cell 3: Run experiments ===
# Option A: just run main.py as-is
# subprocess.run(["python", "-m", "main"], check=True)

# Option B: run specific experiments inline (more flexible for Colab)
from src.experiments import *
from src.experiment_runner import Experiment
from src.utils import (analyze_inference, plot_training_curves, 
                       save_experiment_report, set_random_seed, get_device, 
                       plot_milestone_progress)
from datetime import timedelta
from typing import Dict

def run_single_experiment(config: Dict, exp_name: str, device: torch.device) -> None:
    print(f"--- Starting Experiment: {exp_name} ---")
    set_random_seed(config["seed"])
    
    exp = Experiment(config=config, exp_name=exp_name, device=device)

    train_metrics, train_time = exp.train()
    plot_training_curves(log_dir=exp.results_dir)
    plot_milestone_progress(log_dir=exp.results_dir)

    inference_metrics, inference_time = exp.evaluate()
    analyze_inference(log_dir=exp.results_dir)

    experiment_metrics = train_metrics | inference_metrics 
    timings = {
        "train": str(timedelta(seconds=int(train_time))),
        "inference": str(timedelta(seconds=int(inference_time))),
    }

    save_experiment_report(
        log_dir=exp.results_dir, config=config, 
        metrics=experiment_metrics, timings=timings
    )
    print(f"--- Finished: {exp_name} ---\n")

device = get_device()

# Pick your experiments here:
experiments = [SET4_DDQN_PER_KDB]
for exp in experiments:
    run_single_experiment(exp["config"], exp["name"], device=device)

# === Cell 4: Download results ===
# Zip results folder for easy download from Colab
import shutil
shutil.make_archive("results_export", "zip", "results")
print("Results zipped → results_export.zip")

# In Colab, download with:
# from google.colab import files
# files.download("results_export.zip")