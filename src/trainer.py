import csv
import os
import time
from typing import Dict

from src.agent import BaseAgent
from src.utils import Logger, VideoRecorder


# todo: docstrings
# todo: type hints

def train(env, agent: BaseAgent, logger: Logger, config: Dict) -> None:
    print(f"Starting training: {config['algo']} agent on environment {config['env_name']}")
    
    # init video recorder
    recorder = VideoRecorder(save_dir=logger.log_directory, env=env, agent=agent)

    num_episodes = config['training_episodes']
    for episode in range(1, num_episodes + 1):
        # episode resets:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # todo: consider recording at trianing end as well (last episode)
        record_episode_video = (episode == num_episodes // 2)
        if record_episode_video:
            recorder.start(stage="mid-training")
        
        while not done:
            # capture video frame
            if record_episode_video:
                recorder.capture()

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
        
        # save recorded video
        if record_episode_video:
            recorder.stop()
            print(f"Mid-training video saved to {recorder.filename}")
        
        # determine success
        is_success = (total_reward > 0) # goal reached with positive reward? # todo: change definition?
            
        # log episode metrics + parameters
        logger.log(episode, total_reward, steps, epsilon=0.1, success=is_success)  # todo: pass real epsilon if used
        
        # printout:
        if episode % 1 == 0:# todo change printout frequency
            print(f"Ep {episode} | Reward: {total_reward:.2f} | Steps: {steps}")    # todo: add to printout



def evaluate(env, agent: BaseAgent, logger: Logger, config: Dict, save_dir: str) -> None:
    """
    Runs  inference stage with greedy action (epsilon=0)
    Saves results to inference_log.csv
    """
    num_episodes = config.get("inference_episodes") or 10 # default 10 episodes
    print(f"\nStarting Inference ({num_episodes} episodes)...")
    
    recorder = VideoRecorder(save_dir=logger.log_directory, env=env, agent=agent)

    # init log file
    log_path = os.path.join(save_dir, "inference_log.csv")
    
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success"])
        
        # run evaluation loop
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            if ep == 1:
                recorder.start(stage="post-training")
            
            while not done:
                if ep == 1:
                    recorder.capture()

                # greedy action (no exploration)
                action = agent.choose_action(obs=obs, epsilon=0.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
            
            if ep == 1:
                recorder.stop()
                print(f"Post-training video saved during inference to {recorder.filename}")

            # determine success 
            is_success = (total_reward > 0) # goal reached with positive reward # todo: change definition?
            
            # log episode data
            writer.writerow([ep, total_reward, steps, is_success])

    print(f"Inference data saved to: {log_path}")