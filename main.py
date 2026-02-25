from datetime import timedelta
from typing import Dict

from src.experiments import (
    DQN_SIMPLEGRID_BASELINE, 
    DQN_SIMPLEGRID_STEP_PENALTY, 
    DQN_SIMPLEGRID_STABLE_LOW_LR, 
    DQN_SIMPLEGRID_LONG_EXPLORATION, 
    DQN_KEYDOORBALL_BASELINE, 
    A2C_SIMPLEGRID_BASELINE, 
    A2C_SIMPLEGRID_LOW_ENTROPY, 
    A2C_KEYDOORBALL_BASELINE
)
from src.experiment_runner import Experiment
from src.utils import analyze_inference, plot_training_curves, save_experiment_report, set_random_seed, get_device, plot_milestone_progress


def run_single_experiment(config: Dict, exp_name: str) -> None:
    """Runs one full experiment according to config"""
    print(f"--- Starting Experiment: {exp_name} ---")

    device = get_device()
    set_random_seed(config["seed"])
    
    exp = Experiment(config=config, exp_name=exp_name, device=device)

    # training (note @timer decorator on train(), adds runtime to output)
    train_metrics, train_time = exp.train()
    plot_training_curves(log_dir=exp.results_dir)
    plot_milestone_progress(log_dir=exp.results_dir)

    # inference (note @timer decorator on evaluate(), adds runtime to output)
    inference_metrics, inference_time = exp.evaluate()
    analyze_inference(log_dir=exp.results_dir)

    # collect training + inference metrics
    experiment_metrics = train_metrics | inference_metrics 
    timings = {
        "train": str(timedelta(seconds=int(train_time))),
        "inference": str(timedelta(seconds=int(inference_time))),
    }

    # genereate experiment report
    save_experiment_report(
        log_dir=exp.results_dir, 
        config=config, 
        metrics=experiment_metrics,
        timings=timings
    )
    
    print(f"--- Finished: {exp_name} ---\n")


def main():
    # define exp set:
    experiments = [
        DQN_SIMPLEGRID_STEP_PENALTY,
        DQN_SIMPLEGRID_STABLE_LOW_LR,
        DQN_SIMPLEGRID_LONG_EXPLORATION,
        DQN_SIMPLEGRID_BASELINE,
        A2C_SIMPLEGRID_BASELINE,
        A2C_SIMPLEGRID_LOW_ENTROPY,
        DQN_KEYDOORBALL_BASELINE,
        A2C_KEYDOORBALL_BASELINE,
    ]        

    for exp in experiments:
        run_single_experiment(exp["config"], exp["name"])


if __name__ == "__main__":
    main()
