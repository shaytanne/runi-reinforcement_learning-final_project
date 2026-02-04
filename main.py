import time
from src.utils import analyze_inference, plot_training_curves, set_random_seed, Logger, get_device
from src.agent import BaseAgent, RandomAgent
from src.trainer import train, evaluate
from src.template import SimpleGridEnv, pre_process

def main():
    # configuration
    config = {
        "algo": "TestRun",
        "env_name": "SimpleGrid",
        # "seed": 42,
        "seed": int(time.time()),
        "training_episodes": 5,
        "inference_episodes": 10,
        "obs_shape": (84, 84, 1), 
    }

    # setup infra:
    set_random_seed(seed=config["seed"])
    device = get_device()
    logger = Logger(config=config)
    print(f"Running on: {device}")

    # init environment (from template)
    env = SimpleGridEnv(preprocess=pre_process) 
    
    # init agent
    agent = RandomAgent(
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