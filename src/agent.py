from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.buffer import ReplayBuffer, PrioritizedReplayBuffer
from src.model import ActorCriticNetwork, MiniGridCNN


class BaseAgent(ABC):
    """Base class for agents"""
    gamma: float
    epsilon: float
    learning_rate: float
    device: torch.device
    config: Dict

    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        self.device = device
    
    @abstractmethod
    def choose_action(self, obs, epsilon=0.0) -> int:
        raise NotImplementedError
    
    def step(self, obs, action: int, reward: float, next_obs, done):
        """
        Optional method: stores experience + updates the model in single call (off-policy)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement step(). "
            "This is only needed for per-step update algorithms (e.g. DQN)."
        )

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict:
        """
        Update agent parameters
        DQN: called via step()
        A2C: called with trajectory list
        """
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError
    
    @abstractmethod
    def load(self, filepath: str):
        """Load agent params from checkpoint"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.config.get("algo", "BaseAgent")


class DQNAgent(BaseAgent):
    """
    DQN Agent with Target Network and Replay Buffer.
    """
    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        super().__init__(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
        self.num_actions = num_actions

        self.config = config
        
        # hyperparams
        self.gamma: float = self.config.get("gamma", 0.99)
        self.epsilon: float = self.config.get("epsilon_start", 1.0)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.995)
        self.learning_rate: float = self.config.get("learning_rate", 2.5e-4)
        self.batch_size: int = self.config.get("batch_size", 32)
        self.min_buffer_size: int = self.config.get("min_buffer_size", 1000)
        self.training_freq: int = self.config.get("training_freq", 4)
        self.target_update_freq: int = self.config.get("target_update_freq", 1000)
        buffer_capacity = self.config.get("buffer_capacity", 100_000)

        # linear epsilon decay
        training_episodes = self.config.get("training_episodes", 1000)
        max_steps = self.config.get("max_steps", 200)
        self.epsilon_step: float = (1 - self.epsilon_min) / (0.8 * training_episodes * max_steps) # todo: add 0.8 to config

        # init networks:
        # policy net (main trained network)
        self.policy_net = MiniGridCNN(observation_shape=obs_shape, num_actions=num_actions).to(device)

        # target net (stable reference)
        self.target_net = MiniGridCNN(observation_shape=obs_shape, num_actions=num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # target net never in training mode

        # optimizer + loss # todo: consider other optimizer/loss options?
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate) 
        self.loss_fn = nn.SmoothL1Loss()
        # self.loss_fn = nn.SmoothL1Loss()

        # memory (replay buffer)
        self.memory = ReplayBuffer(capacity=buffer_capacity, obs_shape=obs_shape, device=device)
        
        # init step counter
        self.steps_done = 0

    def choose_action(self, obs: np.ndarray, epsilon: float| None = None) -> int:
        """
        Epsilon-greedy action selection
        :param obs: current observation (np array)
        :param epsilon: exploration probability
        :return: action index
        """
        if epsilon is None:
            epsilon = self.epsilon

        # exploration
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # exploitation
        with torch.no_grad():
            # process state:
            state = torch.as_tensor(data=obs, device=self.device)   # np -> tensor
            state= state.float().div(255.0)                         # normalize (divide by 255)
            state = state.permute(2, 0, 1).unsqueeze(0)             # (height, width, channel) -> (1, channel, height, width)
            
            q_values = self.policy_net(state)   # get Q vals from POLICY net
            best_action = q_values.argmax()     # pick best action
            action_idx = best_action.item()
            return action_idx

    def step(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """
        Execute single step in the env:
        1. store transition
        2. train (if buffer has enough data)
        3. decay epsilon
        """
        self.steps_done += 1
        
        # store step
        self.memory.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        
        # train
        if (len(self.memory) >= self.min_buffer_size) and (self.steps_done % self.training_freq == 0):
            self.update()
            
        # update target net logic:
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # decay epsilon
        self.epsilon = max((self.epsilon * self.epsilon_decay), self.epsilon_min)

    def update(self):
        """
        Core DQN update logic
        """
        # sample batch from buffer
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.memory.sample(self.batch_size)
        
        # calculate current Q values Q(s, a):
        all_q_vals = self.policy_net(state_batch)                       # trigger POLICY net forward, get Q vals
        current_q_vals = all_q_vals.gather(dim=1, index=action_batch)   # filter Q vals for specific action(s) taken 
        
        # calculate Q_target (DOUBLE DQN logic):
        with torch.no_grad():
            # select best action for the next state - use policy net
            best_next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            
            # evaluate Q-value of selected action - use target net
            next_q_vals = self.target_net(next_state_batch).gather(dim=1, index=best_next_actions)
            
            # Bellman: R + gamma * Q(s', argmax Q(s')) * (1 - done)
            target_q_vals = reward_batch + (self.gamma * next_q_vals * (1 - done_batch))
            
        # calculate loss
        loss = self.loss_fn(current_q_vals, target_q_vals)
        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            parameters=self.policy_net.parameters(), 
            max_norm=1.0
        )
        self.optimizer.step()
    
    def save(self, path: str) -> None:
        """
        Save agent snapshot at checkpoint
        - policy net
        - target net
        - optimizer state
        - step counter
        - epsilon
        - config
        """
        agent_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'config': self.config,
        }
        torch.save(agent_dict, path)
        print(f"[DQN] Checkpoint saved to {path}")

    def load(self, filepath: str) -> None:
        """
        Load agent state from a checkpoint file.
        Restores networks, optimizer, step counter, and epsilon so training can resume exactly.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon    = checkpoint['epsilon']
        print(f"[DQN] Checkpoint loaded from {filepath} (step: {self.steps_done}, ε: {self.epsilon:.4f})")


class A2CAgent(BaseAgent):
    """
    A2C (Actor-Critic) Agent
    Components:
    - Actor: learns policy π(a|s) — what action to take
    - Critic: learns value function V(s) — how good is the state
    - Advantage: A(s,a) = R + γ*V(s') - V(s) — how much better is action a vs avg

    Obs format: receives uint8 (H, W, C), normalizes + transposes internally
    No replay buffer — updates once per episode (or other freq)
    """

    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        """
        :param obs_shape:   (H, W, C) — e.g. (84, 84, 1)
        """
        super().__init__(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)

        self.num_actions = num_actions
        self.config = config

        # hyperparams
        self.gamma          = config.get('gamma', 0.99)
        self.learning_rate  = config.get('learning_rate', 3e-4)
        self.value_loss_coefficient = config.get('value_loss_coefficient', 0.5)
        self.entropy_coefficient   = config.get('entropy_coefficient', 0.1)
        self.max_grad_norm  = config.get('max_grad_norm', 0.5)

        # exploitation only (A2C stochastic not epsilon-greedy)
        self.epsilon = 0.0  # attribute kept to adhere to interface

        # step counter
        self.steps_done = 0

        # Actor-Critic network — receives (H, W, C)
        self.network = ActorCriticNetwork(observation_shape=obs_shape, num_actions=num_actions).to(device)

        # single optimizer for both actor and critic heads
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate)

    def choose_action(self, obs: np.ndarray, epsilon: float | None = None) -> int:
        """
        Sample action from current policy π(a|s) (stochastic during training)
        If epsilon=0 (eval mode), act greedily (argmax action probs)
        
        :param obs: uint8 ndarray (H, W, C)
        :param epsilon: ignored during training (A2C explores via policy entropy) # todo
        - pass 0.0 explicitly to get deterministic/greedy inference behaviour
        :return: action index
        """
        with torch.no_grad():
            # ingest observation
            obs_tensor = torch.as_tensor(obs, device=self.device).float()   # uint8 -> float32
            obs_tensor = obs_tensor.div(255.0)                              # normalize to [0,1]
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)           # (H, W, C) -> (1, C, H, W)

            # get action probs from network
            action_logits, _ = self.network(obs_tensor)
            action_probs = F.softmax(action_logits, dim=-1)

            # greedy for eval (epsilon=0.0), stochastic for training
            if epsilon == 0.0:
                return int(action_probs.argmax(dim=-1).item())

            action_dist = torch.distributions.Categorical(action_probs)
            return int(action_dist.sample().item())

    def update(self, trajectories: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> Dict:
        """
        Update actor + critic from full episode (or some freq)

        :param trajectories: list of trajectory data objects (obs, action, reward, next_obs, done)
                          obs, next_obs are uint8 (H, W, C) ndarrays
        :return: update values object (actor_loss, critic_loss, entropy, total_loss, mean_advantage)
        """
        if len(trajectories) == 0:
            return {}

        trajectories = [t[:5] for t in trajectories]    # handle 6th field (PPO interface) # todo
        states, actions, rewards, next_states, dones = zip(*trajectories)

        # states: uint8 (N, H, W, C) -> float32 (N, C, H, W), normalized
        states      = torch.as_tensor(np.array(states),      device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        next_states = torch.as_tensor(np.array(next_states), device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        
        actions     = torch.tensor(actions, dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        # fwd pass thru shared network
        action_logits, state_vals = self.network(states)
        action_probs = F.softmax(action_logits, dim=-1)
        state_vals = state_vals.squeeze()

        # bootstrap next state values from critic
        with torch.no_grad():
            _, next_state_vals = self.network(next_states)
            next_state_vals = next_state_vals.squeeze()

        # TD target: R + γ * V(s') * (1 - done)
        td_target = rewards + self.gamma * next_state_vals * (1 - dones)

        # advantage: A(s,a) = TD target - V(s)
        advantage = td_target - state_vals

        # Actor loss: -log π(a|s) * A(s,a)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        actor_loss = -(log_probs * advantage.detach()).mean()

        # Critic loss: MSE(V(s), TD target)
        critic_loss = F.mse_loss(state_vals, td_target.detach())

        # entropy bonus: encourage exploration (punish overconfident policies)
        entropy = action_dist.entropy().mean()

        # combined loss
        total_loss = (
            actor_loss
            + self.value_loss_coefficient * critic_loss
            - self.entropy_coefficient * entropy
        )

        # optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(parameters=self.network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.steps_done += 1

        return {
            'actor_loss':     actor_loss.item(),
            'critic_loss':    critic_loss.item(),
            'entropy':        entropy.item(),
            'total_loss':     total_loss.item(),
            'mean_advantage': advantage.mean().item(),
        }

    def save(self, path: str) -> None:
        """
        Save agent snapshot at checkpoint
        - network state
        - optimizer state
        - step counter
        - config 
        """
        agent_dict = {
            'network_state_dict':   self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done':           self.steps_done,
            'config':               self.config,
        }
        torch.save(agent_dict, path)
        print(f"[A2C] Checkpoint saved to {path}")

    def load(self, filepath: str) -> None:
        """
        Load network + optimizer state from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        print(f"[A2C] Checkpoint loaded from {filepath} (step: {self.steps_done})")


class PPOAgent(BaseAgent):
    """
    PPO (Proximal Policy Optimization) Agent
    - clipped PPO policy objective
    - GAE for advantage estimation
    - MSE value loss 
    - minibatch/multi-epoch updates
    - clipped PPO policy objective
    """

    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        super().__init__(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)

        self.num_actions = num_actions
        self.config = config
        self.steps_done = 0

        # hyperparams
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.update_epochs = config.get("update_epochs", 4)
        self.value_loss_coefficient = config.get("value_loss_coefficient", 0.5)
        self.entropy_coefficient = config.get("entropy_coefficient", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.minibatch_size = int(config.get("minibatch_size", 64))

        # kept for interface compatibility
        self.epsilon = 0.0

        # shared Actor-Critic network
        self.network = ActorCriticNetwork(
            observation_shape=obs_shape,
            num_actions=num_actions
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )

    def choose_action(self, obs: np.ndarray, epsilon: float | None = None) -> Tuple[int, float]:
        """
        Sample action from current policy π(a|s) + return its log_prob

        - training: sample from categorical distribution
        - eval (epsilon=0.0): greedy argmax
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device).float()
            obs_tensor = obs_tensor.div(255.0)
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)

            action_logits, _ = self.network(obs_tensor)

            if epsilon == 0.0:
                action = int(action_logits.argmax(dim=-1).item())
                return action, 0.0

            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return int(action.item()), float(log_prob.item())

    def update(self, trajectories: List[Tuple[np.ndarray, int, float, np.ndarray, bool, float]]) -> Dict:
        """
        PPO update: run multiple minibatch epochs on one collected rollout.

        Trajectory format:
            (obs, action, reward, next_obs, done, log_prob)

        Notes:
        - "done" here is terminated-only (not truncated), as passed by Experiment.train()
        - that means TD targets still bootstrap across time-limit truncation
        """
        if len(trajectories) == 0:
            return {}

        states, actions, rewards, next_states, dones, old_log_probs = zip(*trajectories)

        # uint8 (N,H,W,C) -> float32 (N,C,H,W), normalized
        states = torch.as_tensor(np.array(states), device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        next_states = torch.as_tensor(np.array(next_states), device=self.device).float().div(255.0).permute(0, 3, 1, 2)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)

        # compute fixed target/advantage once
        with torch.no_grad():
            _, old_values = self.network(states)
            old_values = old_values.squeeze()

            _, next_values = self.network(next_states)
            next_values = next_values.squeeze()

            advantages, returns = self._compute_gae(
                rewards=rewards,
                values=old_values,
                next_values=next_values,
                dones=dones,
            )

            # diagnostics:
            raw_adv_mean = float(advantages.mean().item())
            raw_adv_std = float(advantages.std(unbiased=False).item())
            raw_adv_min = float(advantages.min().item())
            raw_adv_max = float(advantages.max().item())

            returns_mean = float(returns.mean().item())
            returns_std = float(returns.std(unbiased=False).item())

            values_old_mean = float(old_values.mean().item())
            values_old_std = float(old_values.std(unbiased=False).item())

            y = returns
            y_predicted = old_values
            explained_var = 1.0 - torch.var(y - y_predicted) / (torch.var(y) + 1e-8)

        # minibatch PPO epochs
        N = states.shape[0]
        minibatch_size = min(self.minibatch_size, N)

        last_epoch_stats = {}
        for _ in range(self.update_epochs):
            idx = torch.randperm(N, device=self.device)

            approx_kl_epoch = 0.0
            clip_frac_epoch = 0.0
            mb_count = 0
            ratio_mean_sum = 0.0
            ratio_std_sum = 0.0
            v_mean_sum = 0.0
            v_std_sum = 0.0

            for start in range(0, N, minibatch_size):
                mb = idx[start:start + minibatch_size]

                logits, values = self.network(states[mb])
                values = values.squeeze()

                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb])

                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                surr1 = ratio * advantages[mb]

                ratio_clipped = torch.clamp(
                    ratio,
                    min=(1 - self.clip_eps),
                    max=(1 + self.clip_eps)
                )
                surr2 = ratio_clipped * advantages[mb]

                policy_loss = -torch.min(surr1, surr2).mean()

                # safe PPO critic loss: plain MSE (no value clipping)
                value_loss = 0.5 * (values - returns[mb]).pow(2).mean()

                entropy = dist.entropy().mean()

                total_loss = (
                    policy_loss
                    + self.value_loss_coefficient * value_loss
                    - self.entropy_coefficient * entropy
                )

                with torch.no_grad():
                    approx_kl = (old_log_probs[mb] - new_log_probs).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

                    ratio_mean_sum += float(ratio.mean().item())
                    ratio_std_sum += float(ratio.std(unbiased=False).item())
                    v_mean_sum += float(values.mean().item())
                    v_std_sum += float(values.std(unbiased=False).item())
                    approx_kl_epoch += float(approx_kl.item())
                    clip_frac_epoch += float(clip_frac.item())
                    mb_count += 1

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                last_epoch_stats = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "total_loss": float(total_loss.item()),
                    "mean_advantage": float(advantages.mean().item()),
                }

        if mb_count > 0:
            last_epoch_stats["approx_kl_mean"] = approx_kl_epoch / mb_count
            last_epoch_stats["clip_frac_mean"] = clip_frac_epoch / mb_count
            last_epoch_stats["ratio_mean"] = ratio_mean_sum / mb_count
            last_epoch_stats["ratio_std"] = ratio_std_sum / mb_count
            last_epoch_stats["v_mean"] = v_mean_sum / mb_count
            last_epoch_stats["v_std"] = v_std_sum / mb_count

        last_epoch_stats["raw_adv_mean"] = raw_adv_mean
        last_epoch_stats["raw_adv_std"] = raw_adv_std
        last_epoch_stats["raw_adv_min"] = raw_adv_min
        last_epoch_stats["raw_adv_max"] = raw_adv_max
        last_epoch_stats["returns_mean"] = returns_mean
        last_epoch_stats["returns_std"] = returns_std
        last_epoch_stats["values_old_mean"] = values_old_mean
        last_epoch_stats["values_old_std"] = values_old_std
        last_epoch_stats["explained_variance"] = float(explained_var.item())

        self.steps_done += 1
        return last_epoch_stats

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE(λ) advantages & returns
        returns_t = advantages_t + values_t
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = torch.tensor(0.0, device=self.device)

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def save(self, path: str) -> None:
        agent_dict = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "config": self.config,
        }
        torch.save(agent_dict, path)
        print(f"[PPO] Checkpoint saved to {path}")

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]
        print(f"[PPO] Checkpoint loaded from {filepath} (step: {self.steps_done})")


class DDQNPERAgent(BaseAgent):
    """
    Double DQN Agent with PER buffer
    Diff from DQNAgent:
    - uses PrioritizedReplayBuffer (not ReplayBuffer) - weighted sampling by TD error
    - loss weighted by importance-sampling weights (handles non-uniform sampling bias)
    - priorities updated after each training step
    """

    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        super().__init__(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
        self.num_actions = num_actions
        self.config = config

        # hyperparams (same as DQN)
        self.gamma: float = self.config.get("gamma", 0.99)
        self.epsilon: float = self.config.get("epsilon_start", 1.0)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.995)
        self.learning_rate: float = self.config.get("learning_rate", 2.5e-4)
        self.batch_size: int = self.config.get("batch_size", 32)
        self.min_buffer_size: int = self.config.get("min_buffer_size", 1000)
        self.training_freq: int = self.config.get("training_freq", 4)
        self.target_update_freq: int = self.config.get("target_update_freq", 1000)
        buffer_capacity = self.config.get("buffer_capacity", 100_000)

        # linear epsilon decay
        training_episodes = self.config.get("training_episodes", 1000)
        max_steps = self.config.get("max_steps", 200)
        self.epsilon_step: float = (1 - self.epsilon_min) / (0.8 * training_episodes * max_steps)

        # networks (identical to DQN)
        self.policy_net = MiniGridCNN(observation_shape=obs_shape, num_actions=num_actions).to(device)
        self.target_net = MiniGridCNN(observation_shape=obs_shape, num_actions=num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate)

        # PER buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=obs_shape,
            device=device,
            alpha=self.config.get("per_alpha", 0.6),
            beta_start=self.config.get("per_beta_start", 0.4),
            beta_frames=self.config.get("per_beta_frames", 100_000),
            epsilon=self.config.get("per_epsilon", 1e-6),
        )

        self.steps_done = 0

    def choose_action(self, obs: np.ndarray, epsilon: float | None = None) -> int:
        """Epsilon-greedy action selection"""
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state = torch.as_tensor(data=obs, device=self.device)
            state = state.float().div(255.0)
            state = state.permute(2, 0, 1).unsqueeze(0)

            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def step(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """Store transition, train, decay epsilon"""
        self.steps_done += 1

        self.memory.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

        if (len(self.memory) >= self.min_buffer_size) and (self.steps_done % self.training_freq == 0):
            self.update()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max((self.epsilon * self.epsilon_decay), self.epsilon_min)


    def update(self):
        """
        DDQN + PER update logic
        Key diff from DQNAgent:
        1. sample() returns indices + IS weights
        2. loss is element-wise, weighted by IS weights
        3. priorities updated after backprop
        """
        # sample batch - PER has extra fields
        state_batch, next_state_batch, action_batch, reward_batch, done_batch, \
            indices, is_weights = self.memory.sample(self.batch_size)

        # current Q values: Q(s, a)
        all_q_vals = self.policy_net(state_batch)
        current_q_vals = all_q_vals.gather(dim=1, index=action_batch)

        # target Q values (DDQN logic - same as DQNAgent)
        with torch.no_grad():
            best_next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            next_q_vals = self.target_net(next_state_batch).gather(dim=1, index=best_next_actions)
            target_q_vals = reward_batch + (self.gamma * next_q_vals * (1 - done_batch))

        # PER-weighted loss: element-wise TD error * importance-sampling weights
        td_errors = current_q_vals - target_q_vals
        loss = (is_weights * td_errors.pow(2)).mean()

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=self.policy_net.parameters(),
            max_norm=1.0
        )
        self.optimizer.step()

        # update priorities in buffer
        self.memory.update_priorities(
            indices,
            td_errors.detach().cpu().squeeze().numpy()
        )

    def save(self, path: str) -> None:
        agent_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'config': self.config,
        }
        torch.save(agent_dict, path)
        print(f"[DDQN-PER] Checkpoint saved to {path}")

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon    = checkpoint['epsilon']
        print(f"[DDQN-PER] Checkpoint loaded from {filepath} (step: {self.steps_done}, ε: {self.epsilon:.4f})")
