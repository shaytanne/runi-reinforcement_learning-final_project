from abc import ABC, abstractmethod
from typing import Dict
import random

import numpy as np
from pyparsing import Optional
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import MiniGridCNN
from src.buffer import ReplayBuffer
from src.configs import DEFAULT_DQN_CONFIG


class BaseAgent(ABC):
    """Base class for agents"""

    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        self.device = device
    
    @abstractmethod
    def choose_action(self, obs, epsilon=0.0) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def step(self, obs, action: int, reward: float, next_obs, done):
        """Stores experience + updates the model"""
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.config.get("algo", "BaseAgent")


class RandomAgent(BaseAgent):
    """Dummy agent for testing infrastructure"""
    def __init__(self, config, obs_shape, num_actions, device):
        super().__init__(config, obs_shape, num_actions, device)
        self.num_actions = num_actions

    def choose_action(self, obs, epsilon=0.0) -> int:
        return np.random.randint(0, self.num_actions)

    def step(self, obs, action, reward, next_obs, done):
        pass # Do nothing

    def save(self, path):
        pass # Do nothing


class DQNAgent(BaseAgent):
    """
    DQN Agent with Target Network and Replay Buffer.
    """
    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        super().__init__(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
        self.num_actions = num_actions

        self.config = DEFAULT_DQN_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # hyperparams
        self.gamma: float = self.config.get("gamma")
        self.epsilon: float = self.config.get("epsilon_start")
        self.epsilon_min: float = self.config.get("epsilon_min")
        self.epsilon_decay: float = self.config.get("epsilon_decay")
        self.learning_rate: float = self.config.get("learning_rate")
        self.batch_size: int = self.config.get("batch_size")
        self.target_update_freq: int = self.config.get("target_update_freq")

        # init networks:
        # policy net (main trained network)
        self.policy_net = MiniGridCNN(input_shape=obs_shape, num_actions=num_actions).to(device)

        # target net (stable reference)
        self.target_net = MiniGridCNN(input_shape=obs_shape, num_actions=num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # target net never in training mode

        # optimizer + loss # todo: consider other optimizer/loss options?
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate) 
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.SmoothL1Loss()

        # memory (replay buffer)
        buffer_capacity = self.config.get("buffer_capacity") # default to 100,000
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
        # store step
        self.memory.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        
        # train
        if len(self.memory) >= self.batch_size:
            self._learn()
            
        # update target net logic:
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _learn(self):
        """
        Core DQN update logic
        """
        # sample batch from buffer
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.memory.sample(self.batch_size)
        
        # calculate current Q values Q(s, a):
        all_q_vals = self.policy_net(state_batch)                       # trigger POLICY net forward, get Q vals
        current_q_vals = all_q_vals.gather(dim=1, index=action_batch)   # filter Q vals for specific action(s) taken 
        
        # calculate Q_target ( max Q(s', a') , from target net):
        with torch.no_grad():
            #  
            all_next_q_vals = self.target_net(next_state_batch)   # trigger TARGET net forward, get next Q vals
            next_q_vals = all_next_q_vals.max(1)[0].unsqueeze(1)  # filter next Q vals for specific action(s) taken 
            
            # Bellman eq: R + gamma * max(Q(s')) * (1 - done)
            target_q_vals = reward_batch + (self.gamma * next_q_vals * (1 - done_batch))
            
        # calculate loss
        loss = self.loss_fn(current_q_vals, target_q_vals)
        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()

        # todo: necessary?
        # optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            parameters=self.policy_net.parameters(), 
            max_norm=1.0
        )
        self.optimizer.step()

    @property
    def name(self):
        return "DQN"
    
    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)