from datetime import timedelta
import time
from typing import Dict

from src.utils import analyze_inference, plot_training_curves, save_experiment_report, set_random_seed, Logger, get_device
from src.agent import BaseAgent, RandomAgent, DQNAgent
from src.configs import DEFAULT_DQN_CONFIG
from src.trainer import train, evaluate
from src.template import SimpleGridEnv, pre_process


def run_single_experiment(custom_config: Dict, exp_name: str):
    """Runs one full training loop with specific config"""
    
    # setup config
    config = DEFAULT_DQN_CONFIG.copy()
    config.update(custom_config)
    
    # setup logger (append exp_name to folder)
    original_algo_name = config['algo']
    config['algo'] = f"{original_algo_name}_{exp_name}" # todo: better way?
    logger = Logger(config=config)

    device = get_device()
    set_random_seed(config["seed"])
    
    print(f"--- Starting Experiment: {exp_name} ---")
    
    # init env
    env = SimpleGridEnv(preprocess=pre_process, max_steps=200)
    env.reward_config = config.get("reward_config") # injects configurable reward shaping into env # todo 
    
    # init agent
    agent = DQNAgent(
        config=config, 
        obs_shape=config["obs_shape"], 
        num_actions=env.action_space.n, 
        device=device
    )

    # training (note @timer decorator on train(), adds runtime to output)
    train_metrics, train_time = train(env=env, agent=agent, logger=logger, config=config)
    plot_training_curves(logger.log_directory)

    # inference (note @timer decorator on evaluate(), adds runtime to output)
    inference_metrics, inference_time = evaluate(
        env=env, 
        agent=agent, 
        logger=logger, 
        config=config, 
        save_dir=logger.log_directory
    )
    analyze_inference(logger.log_directory)

    # collect training + inference metrics
    experiment_metrics = train_metrics | inference_metrics 
    timings = {
        "train": str(timedelta(seconds=int(train_time))),
        "inference": str(timedelta(seconds=int(inference_time))),
    }

    # genereate experiment report
    save_experiment_report(
        log_dir=logger.log_directory, 
        config=config, 
        metrics=experiment_metrics,
        timings=timings
    )
    
    print(f"--- Finished: {exp_name} ---\n")


def main():
    # define exp set:
    experiments = [
        {
            "name": "Baseline_LR_250u",
            "config": {
                **DEFAULT_DQN_CONFIG,
                "learning_rate": 2.5e-4,
                "env_name": "SimpleGrid",
                "max_steps": 200,
                "training_episodes": 600,
                "inference_episodes": 10,
                "reward_shaping": {
                    "key": 1.0,  
                    "door": 5.0, 
                    "ball": 10.0,
                    "goal": 50.0,
                    "step": -0.01,
                },
            },
        },
        {
            "name": "High_LR_1m",
            "config": {
                **DEFAULT_DQN_CONFIG,
                "learning_rate": 1e-3,
                "env_name": "SimpleGrid",
                "max_steps": 200,
                "training_episodes": 600,
                "inference_episodes": 10,
                "reward_shaping": {
                    "key": 1.0,  
                    "door": 5.0, 
                    "ball": 10.0,
                    "goal": 50.0,
                    "step": -0.01,
                },
            },
        },
    ]        

    for exp in experiments:
        run_single_experiment(exp["config"], exp["name"])


if __name__ == "__main__":
    main()
