from datetime import timedelta
from typing import Dict

from src.experiments import PROJECT_BASE_CONFIG, DQN_SIMPLEGRID_BASELINE, DQN_SIMPLEGRID_STEP_PENALTY, DQN_SIMPLEGRID_STABLE_LOW_LR, DQN_SIMPLEGRID_LONG_EXPLORATION 
from src.utils import analyze_inference, plot_training_curves, save_experiment_report, set_random_seed, get_device
from src.trainer import Experiment


def run_single_experiment(custom_config: Dict, exp_name: str):
    """Runs one full experiment according to config"""
    print(f"--- Starting Experiment: {exp_name} ---")

    # fetch config
    config = PROJECT_BASE_CONFIG.copy()
    config.update(custom_config)
    
    # append exp_name to folder# todo
    original_algo_name = config['algo']
    config['algo'] = f"{original_algo_name}_{exp_name}" # todo: better way?

    device = get_device()
    set_random_seed(config["seed"])
    
    exp = Experiment(config=config, device=device)

    # training (note @timer decorator on train(), adds runtime to output)
    train_metrics, train_time = exp.train()
    plot_training_curves(log_dir=exp.results_dir)

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
        DQN_SIMPLEGRID_BASELINE,
        DQN_SIMPLEGRID_STEP_PENALTY,
        DQN_SIMPLEGRID_STABLE_LOW_LR,
        DQN_SIMPLEGRID_LONG_EXPLORATION,
    ]        

    for exp in experiments:
        run_single_experiment(exp["config"], exp["name"])


if __name__ == "__main__":
    main()
