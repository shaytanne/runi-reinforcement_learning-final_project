import csv
import os
import time
import torch
from typing import Dict

from src.agent import BaseAgent, DQNAgent, RandomAgent
from src.constants import EPISODE_WINDOW_SIZE
from src.template import BaseMiniGridEnv, SimpleGridEnv, KeyDoorBallEnv, pre_process
from src.utils import ExperimentLogger, MetricsHandler, VideoRecorder, timer


class Experiment:
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.training_episodes = config.get("training_episodes", 1000)
        self.inference_episodes = config.get("inference_episodes", 100)

        # init env
        step_limit = config.get("max_steps", 200)
        env_class = self._get_env_class()
        self.env = env_class(preprocess=pre_process, max_steps=step_limit)

        # inject reward config to env
        self.env.reward_shaping = config.get("reward_shaping")
        
        # init agent
        agent_class = self._get_agent_class()
        self.agent = agent_class(
            config=config, 
            obs_shape=config["obs_shape"], 
            num_actions=self.env.action_space.n, 
            device=device
        )

        # experiment results folder
        exp_timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.results_dir = os.path.join("results", f"{config['algo']}_{exp_timestamp}")

        # logger + video recorder
        self.logger = ExperimentLogger(save_dir=self.results_dir)
        self.video_recorder = VideoRecorder(save_dir=self.results_dir, env=self.env)
        
    @timer
    def train(self) -> Dict:
        """
        Runs training loop
        Also handles:
        - video recording
        - logging results
        Returns training metrics object
        """
        print(f"Starting training: {self.agent.name} agent on environment {self.config['env_name']}")

        metrics_handler = MetricsHandler(num_episodes=self.training_episodes, window_size=EPISODE_WINDOW_SIZE)   

        for episode in range(1, self.training_episodes + 1):
            # episode resets:
            obs, _ = self.env.reset()
            done = False
            episode_rewards = 0
            episode_steps = 0

            # todo: record at trianing end as well (last episode)?
            record_episode_video = (episode == self.training_episodes // 2)
            if record_episode_video:
                self.video_recorder.start(stage="mid-training")
            
            while not done:
                # capture video frame
                if record_episode_video: self.video_recorder.capture()

                # take epsilon-greedy  action
                action = self.agent.choose_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # agent step
                self.agent.step(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
                
                # updates:
                episode_rewards += reward
                episode_steps += 1
                obs = next_obs
            
            # save recorded video
            if record_episode_video:
                self.video_recorder.stop()
                print(f"Mid-training video saved to {self.video_recorder.filename}")
            
            # log + print episode metrics
            is_success = (episode_rewards > 0) # goal reached with positive reward # todo: change definition?
            metrics_handler.update(reward=episode_rewards, steps=episode_steps, success=is_success)
            metrics_handler.print_training_status(episode=episode, epsilon=self.agent.epsilon)
            self.logger.log(filename="training_log", 
                            episode=episode, reward=episode_rewards, steps=episode_steps, epsilon=self.agent.epsilon, success=is_success)
        
        # training metrics for whole experiment
        return metrics_handler.get_training_metrics(epsilon=self.agent.epsilon)

    @timer
    def evaluate(self) -> Dict[str, int | float]:
        """
        Runs  inference stage with greedy action (epsilon=0)
        Also handles:
        - video recording
        - logging results
        Returns inference metrics object
        """
        print(f"\nStarting Inference ({self.inference_episodes} episodes)...")

        metrics_handler = MetricsHandler(num_episodes=self.inference_episodes)   

        for episode in range(1, self.inference_episodes + 1):
            # episode resets
            obs, _ = self.env.reset()
            done = False
            episode_rewards = 0
            episode_steps = 0

            if episode == 1: self.video_recorder.start(stage="inference")
            
            while not done:
                # record video of first inference episode (post training)
                if episode == 1: self.video_recorder.capture()
                
                # take greedy action (no exploration)
                action = self.agent.choose_action(obs=obs, epsilon=0.0)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_rewards += reward
                episode_steps += 1
            
            # record video of first inference episode
            if episode == 1:
                self.video_recorder.stop()
                print(f"Post-training video saved during inference to {self.video_recorder.filename}")

            # log episode metrics
            is_success = (episode_rewards > 0) # goal reached with positive reward # todo: change definition?
            metrics_handler.update(reward=episode_rewards, steps=episode_steps, success=is_success)
            self.logger.log(filename="inference_log", 
                            episode=episode, reward=episode_rewards, steps=episode_steps, success=is_success)

        return metrics_handler.get_inference_metrics()

    def _get_env_class(self):
        env_name = self.config.get("env_name")
        if env_name == "SimpleGrid":
            return SimpleGridEnv
        elif env_name == "KeyDoorBall":
            return KeyDoorBallEnv
        else:
            raise ValueError(f"Unknown Environment: {env_name}")

    def _get_agent_class(self):
        algo_name = self.config.get("algo")
        if "DQN" in algo_name: # todo: change to exct name?
            return DQNAgent
        else:
            raise ValueError(f"Unknown Agent Algo: {algo_name}")