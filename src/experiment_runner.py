import os
import time
import torch
from typing import Dict

from src.agent import DQNAgent, DDQNPERAgent, A2CAgent, PPOAgent
from src.constants import EPISODE_WINDOW_SIZE
from src.template import SimpleGridEnv, KeyDoorBallEnv, pre_process
from src.utils import ExperimentLogger, MetricsHandler, VideoRecorder, timer


# --- helpers ---
def _get_milestones(env: KeyDoorBallEnv) -> Dict:
    """
    Reads per-episode milestone flags from KeyDoorBallEnv
    :return: milestone dict for envs that track milestones (e.g. KeyDoorBall), otherwise empty dict (e.g. SimpleGrid)
    """
    if env.__class__.__name__ != "KeyDoorBallEnv":
        return {}
    
    milestones = {attr: int(getattr(env, attr)) for attr in ["has_crossed_door",]
        if hasattr(env, attr)
    }
    milestones |= {
        "got_key":    int(env.is_carrying_key()),
        "opened_door": int(env.is_door_open()),
        "got_ball":   int(env.is_carrying_ball()),
    }
    return milestones


class Experiment:
    def __init__(self, config: Dict, device: torch.device, exp_name: str = ""):
        self.config = config
        self.exp_name = exp_name
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
        agent_class = self._determine_agent_class()
        self.agent = agent_class(
            config=config, 
            obs_shape=config["obs_shape"], 
            num_actions=self.env.action_space.n, 
            device=device
        )

        # experiment results folder
        exp_timestamp = time.strftime('%Y%m%d-%H%M%S')
        exp_name = self.exp_name if self.exp_name else config['algo']
        folder_name = f"{exp_name}_{exp_timestamp}"
        self.results_dir = os.path.join("results", folder_name)

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

        update_per_step = self.config.get("use_per_step_update", True)
        metrics_handler = MetricsHandler(num_episodes=self.training_episodes, window_size=EPISODE_WINDOW_SIZE)   

        for episode in range(1, self.training_episodes + 1):
            # episode resets:
            obs, _ = self.env.reset()
            done = False
            episode_rewards = 0
            episode_steps = 0
            trajectories = []  # for episode-level updates (A2C)
            episode_action_counts = {} # debug

            # todo: record at trianing end as well (last episode)?
            record_episode_video = (episode == self.training_episodes // 2)
            if record_episode_video:
                self.video_recorder.start(stage="mid-training")
            
            while not done:
                # capture video frame
                if record_episode_video: self.video_recorder.capture()

                # select action
                action = self.agent.choose_action(obs)
                log_prob = 0.0  # dummy value for A2C, DQN
                if isinstance(action, tuple):   # handle PPO choose_action output
                    action, log_prob = action

                # monitor action distribution
                episode_action_counts[action] = episode_action_counts.get(action, 0) + 1    # debug

                # env step
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # todo: base this condition on agent type?
                # agent step
                if update_per_step:
                    # DQN: store transition, update, epsilon decay every step
                    self.agent.step(obs=obs, action=action, reward=reward, next_obs=next_obs, done=terminated)
                else:
                    # A2C/PPO: store episode trajectory, update at episode end 
                    # note: log_prob used by PPO, ignored by A2C
                    trajectories.append((obs, action, reward, next_obs, float(terminated), log_prob))

                # updates:
                episode_rewards += reward
                episode_steps += 1
                obs = next_obs
            
            if not update_per_step:
                update_info = self.agent.update(trajectories)

            # save recorded video
            if record_episode_video:
                self.video_recorder.stop()
                print(f"Mid-training video saved to {self.video_recorder.filename}")
            
            # collect milestone state (KeyDoorBall only, no-op for SimpleGrid)
            milestones = _get_milestones(self.env)

            # log + print episode metrics
            is_success = terminated
            metrics_handler.update(reward=episode_rewards, steps=episode_steps, success=is_success)
            metrics_handler.print_training_status(episode=episode, epsilon=self.agent.epsilon)
            self.logger.log(filename="training_log", 
                            episode=episode, reward=episode_rewards, steps=episode_steps, epsilon=self.agent.epsilon, success=is_success,
                            **update_info)
            
            # log action distribution (debug)
            action_dist = {f"action_{a}": episode_action_counts.get(a, 0) for a in range(self.env.action_space.n)}
            self.logger.log(filename="action_dist_training", episode=episode, **action_dist)

            if milestones:
                self.logger.log(filename="milestone_log", episode=episode, **milestones)
        
        # save final model
        self.agent.save(path=os.path.join(self.results_dir, "final_model.pt"))

        # training metrics for whole experiment
        return metrics_handler.get_training_metrics(epsilon=self.agent.epsilon)

    # @timer
    # def evaluate(self) -> Dict[str, int | float]:
    #     """
    #     Runs  inference stage with greedy action (epsilon=0)
    #     Also handles:
    #     - video recording
    #     - logging results
    #     Returns inference metrics object
    #     """
    #     print(f"\nStarting Inference ({self.inference_episodes} episodes)...")

    #     metrics_handler = MetricsHandler(num_episodes=self.inference_episodes)   

    #     for episode in range(1, self.inference_episodes + 1):
    #         # episode resets
    #         obs, _ = self.env.reset()
    #         done = False
    #         episode_rewards = 0
    #         episode_steps = 0

    #         if episode == 1: self.video_recorder.start(stage="inference")
            
    #         while not done:
    #             # record video of first inference episode (post training)
    #             if episode == 1: self.video_recorder.capture()
                
    #             # take greedy action (no exploration)
    #             action = self.agent.choose_action(obs=obs, epsilon=0.0)
    #             if isinstance(action, tuple):   # handle PPO choose_action output
    #                 action = action[0]

    #             # env step
    #             obs, reward, terminated, truncated, _ = self.env.step(action)
    #             done = terminated or truncated
                
    #             episode_rewards += reward
    #             episode_steps += 1
            
    #         # record video of first inference episode
    #         if episode == 1:
    #             self.video_recorder.stop()
    #             print(f"Post-training video saved during inference to {self.video_recorder.filename}")

    #         # log episode metrics
    #         is_success = terminated
    #         metrics_handler.update(reward=episode_rewards, steps=episode_steps, success=is_success)
    #         self.logger.log(filename="inference_log", 
    #                         episode=episode, reward=episode_rewards, steps=episode_steps, success=is_success)

    #     return metrics_handler.get_inference_metrics()

    @timer
    def evaluate(self) -> Dict[str, int | float]:
        """
        Runs inference in two modes:
        1) greedy: epsilon=0.0 (argmax for PPO in your choose_action)
        2) stochastic: sample from policy (PPO) by calling choose_action without epsilon override

        Logs:
        - inference_log_greedy.csv
        - inference_log_stochastic.csv
        Returns:
        - keeps existing 'inference_*' keys for greedy (backwards compatible)
        - adds 'inference_stochastic_*' keys for stochastic
        """
        print(f"\nStarting Inference ({self.inference_episodes} episodes) in 2 modes: greedy + stochastic...")

        # Greedy: keep as primary + record video here (so you still get a deterministic video)
        greedy_metrics = self._run_eval(
            mode_name="greedy",
            epsilon_override=0.0,
            log_filename="inference_log_greedy",
            record_video=True,
        )

        # Stochastic: PPO policy-as-trained (sample). No video by default (set True if you want).
        stochastic_metrics = self._run_eval(
            mode_name="stochastic",
            epsilon_override=None,
            log_filename="inference_log_stochastic",
            record_video=True,
        )

        # Backwards compatible return:
        # - Keep greedy keys as 'inference_*'
        # - Add stochastic keys as 'inference_stochastic_*'
        out = dict(greedy_metrics)
        for k, v in stochastic_metrics.items():
            if k.startswith("inference_"):
                out["inference_stochastic_" + k[len("inference_"):]] = v
            else:
                out["inference_stochastic_" + k] = v

        return out
    
    def _run_eval(self, mode_name: str, epsilon_override, log_filename: str, record_video: bool) -> Dict[str, int | float]:
        metrics_handler = MetricsHandler(num_episodes=self.inference_episodes)

        for episode in range(1, self.inference_episodes + 1):
            obs, _ = self.env.reset()
            done = False
            episode_rewards = 0.0
            episode_steps = 0

            if record_video and episode == 1:
                self.video_recorder.start(stage=f"inference_{mode_name}")

            while not done:
                if record_video and episode == 1:
                    self.video_recorder.capture()

                # --- action selection ---
                if epsilon_override is None:
                    # PPO stochastic: sample from policy distribution
                    action = self.agent.choose_action(obs=obs)
                else:
                    # greedy or explicit epsilon
                    action = self.agent.choose_action(obs=obs, epsilon=epsilon_override)

                if isinstance(action, tuple):  # PPO returns (action, log_prob)
                    action = action[0]

                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_rewards += float(reward)
                episode_steps += 1

            if record_video and episode == 1:
                self.video_recorder.stop()
                print(f"[{mode_name}] video saved to {self.video_recorder.filename}")

            is_success = terminated
            metrics_handler.update(reward=episode_rewards, steps=episode_steps, success=is_success)
            self.logger.log(
                filename=log_filename,
                episode=episode,
                reward=episode_rewards,
                steps=episode_steps,
                success=is_success
            )

        return metrics_handler.get_inference_metrics()

    def _get_env_class(self):
        env_name = self.config.get("env_name")
        if env_name == "SimpleGrid":
            return SimpleGridEnv
        elif env_name == "KeyDoorBall":
            return KeyDoorBallEnv
        else:
            raise ValueError(f"Unknown Environment: {env_name}")

    def _determine_agent_class(self):
        algo_name = self.config.get("algo")
        if algo_name == "DQN":
            return DQNAgent
        elif algo_name == "DDQN_PER":
            return DDQNPERAgent 
        elif algo_name == "A2C":
            return A2CAgent
        elif algo_name == "PPO":
            return PPOAgent
        else:
            raise ValueError(f"Unknown Agent Algo: {algo_name}")