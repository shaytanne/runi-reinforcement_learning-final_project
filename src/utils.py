import os
import csv
import time
import random
from typing import Dict, Optional

import numpy as np
import pandas as pd
import imageio
from matplotlib import pyplot as plt
import torch

from src.agent import BaseAgent


def set_random_seed(seed: int) -> None:
    """
    Sets seed for random number generator
    - cmopatible with CPU/GPU
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_device() -> torch.device:
    """Returns the device that runs training (GPU/CPU)"""
    processor_type = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(processor_type)


# ##########
# PLOTTING UTILS
# ##########
def plot_training_curves(log_dir: str, window: int = 50) -> None:
    """
    Generates single-run convergence graphs with smoothing
    - reqards
    - steps
    - success
    """

    csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    
    # create 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # rewards
    ax1.plot(df['episode'], df['reward'], alpha=0.3, color='tab:blue')
    if len(df) > window:
        ax1.plot(df['episode'], df['reward'].rolling(window).mean(), color='tab:blue', linewidth=2)
    ax1.set_title('Reward')
    ax1.set_xlabel('Episode')

    # steps
    ax2.plot(df['episode'], df['steps'], alpha=0.3, color='tab:orange')
    if len(df) > window:
        ax2.plot(df['episode'], df['steps'].rolling(window).mean(), color='tab:orange', linewidth=2)
    ax2.set_title('Steps to Goal')
    ax2.set_xlabel('Episode')

    # success rate (rolling avg of binary success column)
    if 'success' in df.columns:
        success_data = pd.to_numeric(df['success'], errors='coerce').fillna(0) # cleans col naming/type issues
        if len(df) > window:
            success_data = success_data.rolling(window).mean()

        ax3.plot(df['episode'], success_data, color='tab:green', linewidth=2)
        ax3.set_title(f'Success Rate (Rolling {window})')
        ax3.set_ylim(0, 1.1)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success %')
    else:
        print(f"success column missing, available columns: {df.columns}")
    
    plt.tight_layout()
    save_path = os.path.join(log_dir, "training_curves.png")
    plt.savefig(save_path)
    plt.close()


def plot_comparison(run_directories: Dict, window: int = 50, save_dir: str = "results") -> None:
    """
    Plots multiple runs to compare algorithms    
    :param run_directories: dict like {'DQN': 'results/DQN_.../', 'DoubleDQN': 'results/Double_.../'}
    """

    plt.figure(figsize=(12, 6))
    
    for label, log_dir in run_directories.items():
        csv_path = os.path.join(log_dir, "training_log.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        if len(df) > window:
            # plot only smoothed curve for clarity
            rolling_avg = df['reward'].rolling(window=window).mean()
            plt.plot(df['episode'], rolling_avg, linewidth=2, label=label)
        else:
            plt.plot(df['episode'], df['reward'], linewidth=2, label=label)

    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Algorithm Comparison (Smoothed over {window} eps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "comparison_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def analyze_inference(log_dir: str) -> None:
    """
    Reads inference_log.csv, calculates stats, generates report & plots
    """
    csv_path = os.path.join(log_dir, "inference_log.csv")
    if not os.path.exists(csv_path):
        print(f"No inference log found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        
        # calculate stats:
        avg_reward = df['reward'].mean()
        avg_steps = df['steps'].mean()
        success_rate = df['success'].mean() * 100
        
        print(f"Inference Analysis:")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Avg Steps:    {avg_steps:.1f}")

        # save text report
        report_path = os.path.join(log_dir, "inference_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Inference Analysis ({len(df)} episodes):\n")
            f.write(f"Success Rate: {success_rate:.2f}%\n")
            f.write(f"Avg Reward:   {avg_reward:.4f}\n")
            f.write(f"Avg Steps:    {avg_steps:.2f}\n")
            f.write(f"Std Dev Steps:{df['steps'].std():.2f}\n")

        # histogram plot
        plt.figure(figsize=(8, 6))
        
        # plot only successful episodes for step distribution # todo consider all episodes?
        success_steps = df[df['success'] == 1]['steps']
        if len(success_steps) > 0:
            plt.hist(success_steps, bins=15, color='green', alpha=0.7, label='Successes')
            plt.axvline(avg_steps, color='red', linestyle='dashed', linewidth=2, label=f'Avg Steps: {avg_steps:.1f}')
        else:
            plt.text(0.5, 0.5, "No Successes to Plot", ha='center')

        plt.title(f"Inference: Steps Distribution (Success: {success_rate:.0f}%)")
        plt.xlabel("Steps to Goal")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(log_dir, "inference_histogram.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Inference plots saved to: {save_path}")

    except Exception as e:
        print(f"Error analyzing inference: {e}")
         

# ##########
# LOGGING
# ##########
class Logger:
    """Logging service for the system"""

    def __init__(self, config: Dict):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_directory = os.path.join("results", f"{config['algo']}_{timestamp}") # todo: rename algo
        os.makedirs(self.log_directory, exist_ok=True)
        
        # save config
        config_path = os.path.join(self.log_directory, "config.txt")
        with open(config_path, "w") as f:
            f.write(str(config))

        #  init log file
        self.csv_path = os.path.join(self.log_directory, "training_log.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["episode", "reward", "steps", "epsilon", "success"])

    def log(self, episode: int, reward: float, steps: int, epsilon: float, success: Optional[bool] = None) -> None:
        """Adds single row to log file"""

        row = [episode, reward, steps, epsilon]
        if success is not None: # handle no success info
            row.append(int(success))

        self.writer.writerow(row)
        self.csv_file.flush()


class VideoRecorder:
    """Handles episode video recording, saves to MP4"""
    def __init__(self, save_dir: str, env, agent: BaseAgent, fps: int = 10):
        self.save_dir = save_dir
        self.fps = fps
        self.frames = []
        self.recording = False
        self.filename = None

        self.env = env
        self.agent = agent


    def start(self, stage: str) -> None:
        self.recording = True
        self.frames = []
        env_name = self.env.__class__.__name__
        self.filename = f"{self.agent.name}_{env_name}_{stage}.mp4"

    def capture(self) -> None:
        if self.recording:
            frame = self.env.render() # render mode must be 'rgb_array'
            self.frames.append(frame)

    def stop(self) -> None:
        if self.recording and self.frames:
            path = os.path.join(self.save_dir, self.filename)
            try:
                imageio.mimsave(path, self.frames, fps=self.fps)
                print(f"Video saved: {path}")
            except Exception as e:
                print(f"Video save failed: {e}")
        self.recording = False
        self.frames = []