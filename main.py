import time
from src.utils import analyze_inference, plot_training_curves, set_random_seed, Logger, get_device
from src.agent import BaseAgent, RandomAgent, DQNAgent
from src.configs import DEFAULT_DQN_CONFIG
from src.trainer import train, evaluate
from src.template import SimpleGridEnv, pre_process

def main():
    # experiment configuration:
    config = DEFAULT_DQN_CONFIG.copy()
    config.update({
        "env_name": "SimpleGrid",
        "training_episodes": 600,
        "inference_episodes": 10,
        "buffer_capacity": 100000,   # replay buffer for DQN 
    })

    # setup infra:
    set_random_seed(seed=config["seed"])
    device = get_device()
    logger = Logger(config=config)
    print(f"Running on: {device}")

    # init environment (from template)
    env = SimpleGridEnv(preprocess=pre_process) 
    
    # init agent
    agent = DQNAgent(
        config=config, 
        obs_shape=config["obs_shape"], 
        num_actions=env.action_space.n, # todo: is this available?
        device=device
    )

    # run training
    train(env=env, agent=agent, logger=logger, config=config)
    plot_training_curves(log_dir=logger.log_directory)

    # run inference
    evaluate(
        env=env, 
        agent=agent,
        logger=logger,
        config=config, 
        save_dir=logger.log_directory, 
    )
    analyze_inference(logger.log_directory)

if __name__ == "__main__":
    main()