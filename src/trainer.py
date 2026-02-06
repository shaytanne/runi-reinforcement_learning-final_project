from collections import deque
import csv
import os
import time
from typing import Dict

from src.agent import BaseAgent
from src.constants import EPISODE_WINDOW_SIZE
from src.template import BaseMiniGridEnv
from src.utils import Logger, VideoRecorder, timer


@timer
def train(env: BaseMiniGridEnv, agent: BaseAgent, logger: Logger, config: Dict) -> Dict:
    """
    Runs training loop
    Also handles:
    - video recording
    - logging results
    Returns training metrics object
    """
    print(f"Starting training: {config['algo']} agent on environment {config['env_name']}")
    
    # init video recorder
    recorder = VideoRecorder(save_dir=logger.log_directory, env=env, agent=agent)

    # tracked metrics:
    reward_window = deque(maxlen=EPISODE_WINDOW_SIZE)
    steps_window = deque(maxlen=EPISODE_WINDOW_SIZE)
    success_window = deque(maxlen=EPISODE_WINDOW_SIZE)

    num_episodes = config['training_episodes']
    for episode in range(1, num_episodes + 1):
        # episode resets:
        obs, _ = env.reset()
        done = False
        episode_rewards = 0
        episode_steps = 0

        # todo: record at trianing end as well (last episode)?
        record_episode_video = (episode == num_episodes // 2)
        if record_episode_video:
            recorder.start(stage="mid-training")
        
        while not done:
            # capture video frame
            if record_episode_video: recorder.capture()

            # take epsilon-greedy  action
            action = agent.choose_action(obs)
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
            episode_rewards += reward
            episode_steps += 1
            obs = next_obs
        
        # save recorded video
        if record_episode_video:
            recorder.stop()
            print(f"Mid-training video saved to {recorder.filename}")
        
        # episode metrics
        is_success = (episode_rewards > 0) # goal reached with positive reward # todo: change definition?
        success_window.append(int(is_success))
        reward_window.append(episode_rewards)
        steps_window.append(episode_steps)
            
        # log episode metrics/parameters
        logger.log(episode, episode_rewards, episode_steps, epsilon=agent.epsilon, success=is_success)  # todo: pass real epsilon if used
        
        # printout:
        if episode % EPISODE_WINDOW_SIZE == 0:
            avg_reward = sum(reward_window) / len(reward_window)
            avg_steps = sum(steps_window) / len(steps_window)
            success_rate = sum(success_window) / len(success_window)
            print(f"Episodes {episode-EPISODE_WINDOW_SIZE}-{episode}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | Avg Steps: {avg_steps:.2f} | Success Rate: {success_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    experiment_metrics = {
        "train_episodes": num_episodes,
        "train_final_epsilon": agent.epsilon,
        "train_window_success_rate": sum(success_window) / len(success_window) if success_window else 0.0,
        "train_window_avg_reward": sum(reward_window) / len(reward_window) if reward_window else 0.0,
        "train_window_avg_steps": sum(steps_window) / len(steps_window) if steps_window else 0.0
    }
    return experiment_metrics


@timer
def evaluate(env: BaseMiniGridEnv, agent: BaseAgent, logger: Logger, config: Dict, save_dir: str) -> Dict[str, int | float]:
    """
    Runs  inference stage with greedy action (epsilon=0)
    Also handles:
    - video recording
    - logging results
    Returns inference metrics object
    """
    num_episodes: int = config.get("inference_episodes")
    print(f"\nStarting Inference ({num_episodes} episodes)...")
    
    recorder = VideoRecorder(save_dir=logger.log_directory, env=env, agent=agent)

    # init log file
    log_path = os.path.join(save_dir, "inference_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success"])
        
        # experiment metrics
        success_count = 0
        total_rewards = 0
        step_counts = []
        success_step_counts = [] # for successful episodes only

        # evaluation loop
        for episode in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            episode_rewards = 0
            episode_steps = 0

            if episode == 1:
                recorder.start(stage="post-training")
            
            while not done:
                # record video of first inference episode
                if episode == 1: recorder.capture()

                # take greedy action (no exploration)
                action = agent.choose_action(obs=obs, epsilon=0.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_rewards += reward
                episode_steps += 1
            
            # record video of first inference episode
            if episode == 1:
                recorder.stop()
                print(f"Post-training video saved during inference to {recorder.filename}")

            # determine success 
            is_success = (episode_rewards > 0) # goal reached with positive reward # todo: change definition?
            
            # log episode metrics
            writer.writerow([episode, episode_rewards, episode_steps, is_success])

            # update experiment metrics
            success_count += int(is_success)
            total_rewards += episode_rewards
            step_counts.append(episode_steps)
            if is_success:
                success_step_counts.append(episode_steps)

    print(f"Inference data saved to: {log_path}")
    experiment_metrics = {
        "inference_episodes": num_episodes,
        "inference_success_rate": success_count / num_episodes,
        "inference_avg_reward": total_rewards / num_episodes,
        "inference_avg_steps": sum(step_counts) / len(step_counts) if step_counts else 0,
        "inference_avg_steps_to_success": (sum(success_step_counts) / len(success_step_counts)) if success_step_counts else 0,
    }
    return experiment_metrics