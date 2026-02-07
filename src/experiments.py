import copy
import time


# standard/fallback exp config, basis for all agents/envs/setups
PROJECT_BASE_CONFIG = {
    # run settings (env, algorithm, setup)
    "env_name": "SimpleGrid",
    "algo": "DQN",              # which agent
    "obs_shape": (84, 84, 1),
    "seed": int(time.time()),   # random seed
    "max_steps": 200,           # per episode
    "training_episodes": 1000,
    "inference_episodes": 20,
    
    # hyperparameters
    "gamma": 0.99,              # discount factor
    "learning_rate": 2.5e-4,    # learning rate 
    "epsilon_start": 1.0,       # initial epsilon
    "epsilon_min": 0.05,        # minimum epsilon
    "epsilon_decay": 0.995,     # epsilon decay rate
    "batch_size": 32,           # batch size
    "buffer_capacity": 100000,  # replay buffer capacity
    "training_freq": 4,         # train (backprop) every N(=4) steps
    "target_update_freq": 1000, # sync target network every N(=1000) steps
    "grad_clip": 1.0,           # gradient clipping value
    
    # reward shaping
    "reward_shaping": {
        "step": 0.0,
        "goal": 1.0
    }
}

# 1. baseline
DQN_SIMPLEGRID_BASELINE = {
    "name": "1_Baseline",
    "config": copy.deepcopy(PROJECT_BASE_CONFIG),
}

# 2. reward shaoing: step penalty
DQN_SIMPLEGRID_STEP_PENALTY = {
    "name": "2_Step_Penalty",
    "config": copy.deepcopy(PROJECT_BASE_CONFIG),
}
DQN_SIMPLEGRID_STEP_PENALTY["config"]["reward_shaping"] = {"step": 0.01, "goal": 1.0}

# 3. stability focus: low LR, slower target updates
DQN_SIMPLEGRID_STABLE_LOW_LR = {
    "name": "3_Stable_LowLR",
    "config": copy.deepcopy(PROJECT_BASE_CONFIG),
}
DQN_SIMPLEGRID_STABLE_LOW_LR["config"]["learning_rate"] = 1e-4
DQN_SIMPLEGRID_STABLE_LOW_LR["config"]["target_update_freq"] = 2000

# 4. long exploration (epsilon: slow decay, lower min)
DQN_SIMPLEGRID_LONG_EXPLORATION = {
    "name": "4_Long_Exploration",
    "config": copy.deepcopy(PROJECT_BASE_CONFIG),
}
DQN_SIMPLEGRID_LONG_EXPLORATION["config"]["epsilon_decay"] = 0.999
DQN_SIMPLEGRID_LONG_EXPLORATION["config"]["epsilon_min"] = 0.1