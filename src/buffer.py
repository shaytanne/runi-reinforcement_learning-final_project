from typing import Tuple
import numpy as np
import torch
from torch import Tensor

from src.sum_tree import SumTree

class ReplayBuffer:
    """
    Replay buffer class for use by agents during training
    """

    def __init__(self, capacity: int, obs_shape: np.ndarray, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.occupancy = 0  # number of stored transitions
        self.index = 0      # next index to store transition

        # init stoaage arrays
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.action = np.zeros((capacity, 1), dtype=np.int64)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.done   = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """Store a transition in the buffer"""
        
        self.obs[self.index] = obs
        self.next_obs[self.index] = next_obs
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.done[self.index] = done

        # update index and occupancy
        self.index = (self.index + 1) % self.capacity
        self.occupancy = min(self.occupancy + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample a random batch of transitions from the buffer"""
        idxs = np.random.randint(0, self.occupancy, size=batch_size)

        # sample + normalize (divide by 255) + handle shape convesions
        obs_sample = torch.as_tensor(self.obs[idxs], device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        next_obs_sample = torch.as_tensor(self.next_obs[idxs], device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        actions_sample = torch.as_tensor(self.action[idxs], device=self.device).long()
        reward_sample = torch.as_tensor(self.reward[idxs], device=self.device).float()
        done_sample = torch.as_tensor(self.done[idxs], device=self.device).float()

        return (
            obs_sample, 
            next_obs_sample, 
            actions_sample, 
            reward_sample, 
            done_sample
        )
    
    def __len__(self):
        return self.occupancy


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer
    Samples transitions proportional to their TD-error priority
    Uses SumTree for O(log N) sampling (no sorting)
    """

    def __init__(self, capacity: int, obs_shape: np.ndarray, device: torch.device, alpha: float = 0.6, 
                 beta_start: float = 0.4, beta_frames: int = 100_000, epsilon: float = 1e-6):
        self.capacity = capacity
        self.device = device
        self.occupancy = 0
        self.index = 0

        # PER hyperparameters
        self.alpha = alpha              # priority exponent: 0 = uniform, 1 = full prioritization
        self.beta_start = beta_start    # init importance-sampling correction
        self.beta_frames = beta_frames  # num steps for beta decay to 1.0
        self.epsilon = epsilon          # prevents 0 priority
        self.frame = 0                  # step counter for beta decay

        # storage arrays (same as ReplayBuffer)
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.action   = np.zeros((capacity, 1), dtype=np.int64)
        self.reward   = np.zeros((capacity, 1), dtype=np.float32)
        self.done     = np.zeros((capacity, 1), dtype=np.float32)

        # sumtree for priority-weighted sampling
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        """Decay beta linearly from beta_start to 1.0 over <beta_frames> steps"""
        fraction = min(self.frame / max(self.beta_frames, 1), 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """Store transition with max priority so new experiences get replayed at least once."""
        self.obs[self.index] = obs
        self.next_obs[self.index] = next_obs
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.done[self.index] = done

        # new transitions get max priority
        self.tree.add(self.max_priority ** self.alpha)

        self.index = (self.index + 1) % self.capacity
        self.occupancy = min(self.occupancy + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, np.ndarray, Tensor]:
        """
        Sample batch proportional to priorities
        :return: buffer item (obs, next_obs, actions, rewards, dones, idxs, is_weights)
        """
        self.frame += 1
        idxs = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float32)

        # divide total priority into equal segments
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx = self.tree.sample(value)
            idxs[i] = idx
            priorities[i] = max(self.tree.tree[idx + self.tree.capacity], self.epsilon)

        # importance-sampling weights: wi = (N * P(i))^(-beta), normalized
        probs = priorities / self.tree.total
        probs = np.clip(probs, self.epsilon, None)  # prevent 0 probabilities
        beta = self.beta
        is_weights = (self.occupancy * probs) ** (-beta)
        is_weights /= is_weights.max()  # normalize so max weight=1
        is_weights = torch.as_tensor(is_weights, device=self.device).float().unsqueeze(1)

        # fetch transitions (same as ReplayBuffer)
        obs_sample      = torch.as_tensor(self.obs[idxs],      device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        next_obs_sample = torch.as_tensor(self.next_obs[idxs], device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        actions_sample  = torch.as_tensor(self.action[idxs],   device=self.device).long()
        reward_sample   = torch.as_tensor(self.reward[idxs],   device=self.device).float()
        done_sample     = torch.as_tensor(self.done[idxs],     device=self.device).float()

        return (
            obs_sample,
            next_obs_sample,
            actions_sample,
            reward_sample,
            done_sample,
            idxs,
            is_weights,
        )

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors from last training step"""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self):
        return self.occupancy