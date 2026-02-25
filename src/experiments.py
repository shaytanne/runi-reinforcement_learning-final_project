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
    "min_buffer_size": 1000,    # minimum buffer size before training
    "training_freq": 4,         # train (backprop) every N(=4) steps
    "target_update_freq": 1000, # sync target network every N(=1000) steps
    "grad_clip": 1.0,           # gradient clipping value for DQN

    # A2C/PPO hyperparameters (ignored by DQN)
    "value_loss_coefficient": 0.5,     # weight of critic loss in total loss
    "entropy_coefficient": 0.01,       # weight of entropy bonus (encourages exploration)
    "max_grad_norm": 0.5,              # grad clipping for A2C/PPO
    "use_per_step_update": True,       # True = DQN (step-level)  False = A2C/PPO (episode-level)

    # PPO-specific (ignored by DQN/A2C)
    "clip_eps": 0.2,
    "gae_lambda": 0.95, # todo
    "update_epochs": 4,
    
    # reward shaping
    "reward_shaping": {
        "step": 0.0,
        "goal": 1.0
    }
}

# A2C base config (override DQN-specific fields, keep shared ones)
A2C_BASE_CONFIG = copy.deepcopy(PROJECT_BASE_CONFIG)
A2C_BASE_CONFIG.update({
    "algo": "A2C",
    "use_per_step_update": False,   # update per episode not step

    # hyperparams adjusted for A2C 
    "learning_rate": 3e-4,  # higher LR than DQN
    "gamma": 0.99,
    "value_loss_coefficient": 0.5,
    "entropy_coefficient": 0.1,
    "max_grad_norm": 0.5,

    # disable fields unused by A2C
    "epsilon_start": None,
    "epsilon_min": None,
    "epsilon_decay": None,
    "batch_size": None,
    "buffer_capacity": None,
    "min_buffer_size": None,
    "training_freq": None,
    "target_update_freq": None,
})

PPO_BASE_CONFIG = copy.deepcopy(PROJECT_BASE_CONFIG)
PPO_BASE_CONFIG.update({
    "algo": "PPO",
    "use_per_step_update": False,   # episode-level update (same as A2C)
    "learning_rate": 3e-4,
    "value_loss_coefficient": 0.5,
    "entropy_coefficient": 0.01,
    "max_grad_norm": 0.5,
    "clip_eps": 0.2,
    "gae_lambda": 0.95,
    "update_epochs": 4,

    # disable DQN-only fields
    "epsilon_start": None,
    "epsilon_min": None,
    "epsilon_decay": None,
    "batch_size": None,
    "buffer_capacity": None,
    "min_buffer_size": None,
    "training_freq": None,
    "target_update_freq": None,
})

# =====================================================================
#                          DQN on SimpleGrid
# =====================================================================

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


# =====================================================================
#                          A2C on SimpleGrid
# =====================================================================

# 5. A2C baseline on SimpleGrid
A2C_SIMPLEGRID_BASELINE = {
    "name": "5_A2C_Baseline",
    "config": copy.deepcopy(A2C_BASE_CONFIG),
}

# 6. A2C, lower entropy (more exploitation)
A2C_SIMPLEGRID_LOW_ENTROPY = {
    "name": "6_A2C_Low_Entropy",
    "config": copy.deepcopy(A2C_BASE_CONFIG),
}
A2C_SIMPLEGRID_LOW_ENTROPY["config"]["entropy_coefficient"] = 0.01


# =====================================================================
#                          PPO on SimpleGrid
# =====================================================================

# 7. PPO baseline on SimpleGrid
PPO_SIMPLEGRID_BASELINE = {
    "name": "7_PPO_Baseline",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}


# =====================================================================
#                          DQN on KeyDoorBall
# =====================================================================

# 8. DQN baseline on KeyDoorBall
DQN_KEYDOORBALL_BASELINE = {
    "name": "8_DQN_KeyDoorBall",
    "config": copy.deepcopy(PROJECT_BASE_CONFIG),
}
DQN_KEYDOORBALL_BASELINE["config"].update({
    "env_name": "KeyDoorBall",
    "max_steps": 500,
    "training_episodes": 5000,
    "inference_episodes": 10,
    "reward_shaping": {
        "key":           0.5,
        "door":          0.5,
        "room_crossing": 1.0,
        "ball":          0.5,
        "goal":          2.0,
        "turn_penalty":  0.1,
        "step":          0.001,
    }
})


# =====================================================================
#                          A2C on KeyDoorBall
# =====================================================================

# 9. A2C baseline on KeyDoorBall: longer episodes, more training
A2C_KEYDOORBALL_BASELINE = {
    "name": "9_A2C_KeyDoorBall",
    "config": copy.deepcopy(A2C_BASE_CONFIG),
}
A2C_KEYDOORBALL_BASELINE["config"].update({
    "env_name": "KeyDoorBall",
    "max_steps": 500,
    "training_episodes": 5000,
    "inference_episodes": 10,
    "reward_shaping": {
        "key":          0.5,
        "door":         0.5,
        "room_crossing": 1.0,
        "ball":         0.5,
        "goal":         2.0,
        "turn_penalty": 0.1,
        "step":         0.001,
    },
})


# =====================================================================
#                          PPO on KeyDoorBall
# =====================================================================

# 10. PPO baseline on KeyDoorBall
PPO_KEYDOORBALL_BASELINE = {
    "name": "10_PPO_KeyDoorBall",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
PPO_KEYDOORBALL_BASELINE["config"].update({
    "env_name": "KeyDoorBall",
    "max_steps": 500,
    "training_episodes": 5000,
    "inference_episodes": 10,
    "reward_shaping": {
        "key":           0.5,
        "door":          0.5,
        "room_crossing": 1.0,
        "ball":          0.5,
        "goal":          2.0,
        "turn_penalty":  0.1,
        "step":          0.001,
    },
})