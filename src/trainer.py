import csv
import os
import time
from typing import Dict

from src.agent import BaseAgent
from src.utils import Logger


# todo: docstrings
# todo: type hints

def train(env, agent: BaseAgent, logger: Logger, config: Dict):
    print(f"Starting training: {config['algo']} agent on environment {config['env_name']}")
    
    num_episodes = config['episodes']
    for episode in range(1, num_episodes + 1):
        # episode resets:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # pick action
            action = agent.choose_action(obs)
            
            # env step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # agent step
            agent.step(
                obs=obs, 
                action=action, 
                reward=reward, 
                next_obs=next_obs, 
                done=done
            )
            
            # updates:
            total_reward += reward
            steps += 1
            obs = next_obs
            
        # log episode metrics + parameters
        logger.log(episode, total_reward, steps, epsilon=0.1)
        
        # printout:
        if episode % 1 == 0:# todo change printout frequency
            print(f"Ep {episode} | Reward: {total_reward:.2f} | Steps: {steps}")    # todo: add to printout



def evaluate(env, agent: BaseAgent, config: Dict, save_dir: str, num_eval_episodes: int = 100) -> None:
    """
    Runs  inference stage with greedy action (epsilon=0)
    Saves results to inference_log.csv
    """
    print(f"\nStarting Inference Evaluation ({num_eval_episodes} episodes)...")
    
    # init log file
    log_path = os.path.join(save_dir, "inference_log.csv")
    
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success"])
        
        # run evaluation loop
        for ep in range(1, num_eval_episodes + 1):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # greedy action (no exploration)
                action = agent.choose_action(obs=obs, epsilon=0.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
            
            # determine success 
            is_success = 1 if total_reward > 0 else 0 # goal reached with positive reward # todo: change definition?
            
            # log episode data
            writer.writerow([ep, total_reward, steps, is_success])

    print(f"Inference data saved to: {log_path}")