import copy
import time


COMMON_BASE_CONFIG = {
    # environment
    "env_name":             "SimpleGrid",
    "obs_shape":            (84, 84, 1),
    "max_steps":            200,
    "seed":                 int(time.time()),

    # training loop
    "training_episodes":    1000,
    "inference_episodes":   20,

    # shared hyperparameters
    "gamma":                0.99,

    # reward shaping
    "reward_shaping": {
        "step": 0.0,
        "goal": 1.0,
    },
}

DQN_BASE_CONFIG = copy.deepcopy(COMMON_BASE_CONFIG)
DQN_BASE_CONFIG.update({
    "algo":                 "DQN",
    "use_per_step_update":  True,

    # network + optimizer
    "learning_rate":        2.5e-4,
    "grad_clip":            1.0,

    # exploration
    "epsilon_start":        1.0,
    "epsilon_min":          0.05,
    "epsilon_decay":        0.995,

    # replay buffer
    "batch_size":           128,
    "buffer_capacity":      100_000,
    "min_buffer_size":      1000,

    # update schedule
    "training_freq":        4,
    "target_update_freq":   1000,
})

A2C_BASE_CONFIG = copy.deepcopy(COMMON_BASE_CONFIG)
A2C_BASE_CONFIG.update({
    "algo":                     "A2C",
    "use_per_step_update":      False,

    # network + optimizer
    "learning_rate":            3e-4,
    "max_grad_norm":            0.5,

    # loss coefficients
    "value_loss_coefficient":   0.5,
    "entropy_coefficient":      0.1,
})

PPO_BASE_CONFIG = copy.deepcopy(A2C_BASE_CONFIG)   # ← inherits from A2C, not SHARED
PPO_BASE_CONFIG.update({
    "algo":                     "PPO",

    # PPO-specific
    "entropy_coefficient":      0.01,   # PPO uses lower entropy than A2C
    "clip_eps":                 0.2,
    "gae_lambda":               0.95,
    "update_epochs":            4,
})


# =====================================================================
#                       SET 2
# =====================================================================
SET2_MAX_STEPS = 800
SET2_EPISODES = 500
SET2_INFERENCE_EPISODES = 50

# PPO on KDB
SET2_PPO_KDB = {
    "name": "SET2_PPO_KDB",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
SET2_PPO_KDB["config"].update({
    "env_name": "KeyDoorBall",
    "training_episodes": SET2_EPISODES,
    "inference_episodes": SET2_INFERENCE_EPISODES,
    "max_steps": SET2_MAX_STEPS,
    "reward_shaping": {
        "key": 0.5, "door": 0.9, "room_crossing": 1.2,
        "ball": 1.5, "goal": 2.0, "turn_penalty": 0.0, "step": 0.001,
    },
    "training_freq":    10,        # train (backprop) every N(=10) steps
})

# DQN on KDB
SET2_DQN_KDB = {
    "name": "SET2_DQN_KDB",
    "config": copy.deepcopy(DQN_BASE_CONFIG),
}
SET2_DQN_KDB["config"].update({
    "env_name": "KeyDoorBall",
    "training_episodes": SET2_EPISODES,
    "inference_episodes": SET2_INFERENCE_EPISODES,
    "max_steps": SET2_MAX_STEPS,
    "reward_shaping": {
        "key": 0.5, "door": 0.9, "room_crossing": 1.2,
        "ball": 1.5, "goal": 2.0, "turn_penalty": 0.0, "step": 0.001,
    },
    "training_freq":    10,        # train (backprop) every N(=10) steps
})

exp_set_2 = [
    SET2_PPO_KDB,
    SET2_DQN_KDB,
]

# =====================================================================
#                       SET 3
# =====================================================================
SET3_MAX_STEPS = 250
SET3_EPISODES = 3000
SET3_INFERENCE_EPISODES = 50

# PPO on KDB
SET3_PPO_KDB = {
    "name": "SET3_PPO_KDB",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
SET3_PPO_KDB["config"].update({
    "env_name": "KeyDoorBall",
    "training_episodes": SET3_EPISODES,
    "inference_episodes": SET3_INFERENCE_EPISODES,
    "max_steps": SET3_MAX_STEPS,
    "epsilon_decay":        0.999,
    "training_freq":    10,        # train (backprop) every N(=10) steps
    "reward_shaping": {
        "key": 0.5, "door": 0.9, "room_crossing": 1.2,
        "ball": 1.5, "goal": 2.0, "turn_penalty": 0.0, "step": 0.001,
    },
})

# DQN on KDB
SET3_DQN_KDB = {
    "name": "SET3_DQN_KDB",
    "config": copy.deepcopy(DQN_BASE_CONFIG),
}
SET3_DQN_KDB["config"].update({
    "env_name": "KeyDoorBall",
    "training_episodes": SET3_EPISODES,
    "inference_episodes": SET3_INFERENCE_EPISODES,
    "max_steps": SET3_MAX_STEPS,
    "epsilon_decay":        0.999,
    "training_freq":    10,        # train (backprop) every N(=10) steps
    "reward_shaping": {
        "key": 0.5, "door": 0.9, "room_crossing": 1.2,
        "ball": 1.5, "goal": 2.0, "turn_penalty": 0.0, "step": 0.001,
    },
    
})


SET3_DQN_KDB_LINEAR_EPSILON = {
    "name": "SET3_DQN_KDB_LinearEpsilon",
    "config": copy.deepcopy(DQN_BASE_CONFIG),
}
SET3_DQN_KDB_LINEAR_EPSILON["config"].update({
    "env_name": "KeyDoorBall",
    "training_episodes": SET3_EPISODES,
    "inference_episodes": SET3_INFERENCE_EPISODES,
    "max_steps": SET3_MAX_STEPS,
    "epsilon_decay":        0.999,
    "training_freq":    4,        # train (backprop) every N(=10) steps
    "reward_shaping": {
        "key": 5.0, "door": 10.0, "room_crossing": 10.0,
        "ball": 10.0, "goal": 40.0, "turn_penalty": 0.01, "step": 0.001,
    },
    
})

exp_set_3 = [
    SET3_DQN_KDB,
    SET3_PPO_KDB, 
]
