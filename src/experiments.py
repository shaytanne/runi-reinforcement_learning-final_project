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
    "batch_size":           256,
    "buffer_capacity":      40_000,
    "min_buffer_size":      5000,

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
SET3_MAX_STEPS = 450
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
    "epsilon_min": 0.1,
    # "epsilon_decay":        0.999,
    "training_freq":    4,        # train (backprop) every N(=10) steps
    "reward_shaping": {
        "key": 1.0, "door": 1.0, "room_crossing": 1.0,
        "ball": 1.5, "goal": 2.0, "turn_penalty": 0.0, "step": 0.001,
    },
    
})

exp_set_3 = [
    SET3_DQN_KDB,
    SET3_PPO_KDB, 
]


# =====================================================================
#   exp set 4 DDQN + PER 
# =====================================================================
DDQN_PER_BASE_CONFIG = copy.deepcopy(DQN_BASE_CONFIG)
DDQN_PER_BASE_CONFIG.update({
    "algo":                 "DDQN_PER",
    "use_per_step_update":  True,

    # PER hyperparameters
    "per_alpha":            0.6,        # priority exponent (0=uniform, 1=full priority)
    "per_beta_start":       0.4,        # init IS correction (decays to 1.0)
    "per_beta_frames":      100_000,    # num steps to decay beta
    "per_epsilon":          1e-6,       # prevent 0-priority
})


SET4_DDQN_PER_KDB = {
    "name": "SET4_DDQN_PER_KDB",
    "config": copy.deepcopy(DDQN_PER_BASE_CONFIG),
}
SET4_DDQN_PER_KDB["config"].update({
    "env_name":             "KeyDoorBall",
    "obs_shape":            (84, 84, 1), # grayscale
    "training_episodes":    3000,
    "inference_episodes":   50,
    "max_steps":            450,

    # buffer
    "buffer_capacity":      400_000,
    "min_buffer_size":      20_000,
    "batch_size":           256,

    # exploration
    "epsilon_decay":        0.9977, 
    "epsilon_min":          0.1, 

    # update schedule
    "training_freq":        4,
    "target_update_freq":   2000,

    # PER params
    "per_alpha":            0.6,
    "per_beta_start":       0.4,
    "per_beta_frames":      600_000,             # assumes 3000 eps x 450 steps x 0.44

    # reward shaping
    "reward_shaping": {
        "key": 0.5, "door": 1.0, "room_crossing": 1.5,
        "ball": 2.0, "goal": 3.0, "turn_penalty": 0.0, "step": 0.001,
    },
})


SET4_PPO_KDB = {
    "name": "SET4_PPO_KDB",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
SET4_PPO_KDB["config"].update({
    "env_name": "KeyDoorBall",
    "obs_shape": (84, 84, 1),
    "max_steps": 450,
    "seed": 1772571932,
    "training_episodes": 3000,
    "inference_episodes": 50,
    "minibatch_size": 64,
    "reward_shaping": {
        "key": 1.0,
        "door": 2.0,
        "room_crossing": 3,
        "ball": 4.0,
        "goal": 12.0,
        "turn_penalty": 0.0,
        "step": 0.001,
    },
})

SET4_PPO_FINE_TUNE = {
    "name": "SET4_PPO_Fine_Tune",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
SET4_PPO_FINE_TUNE["config"].update({
    "env_name": "KeyDoorBall",
    "obs_shape": (84, 84, 1),
    "max_steps": 450,
    "seed": 1772571932,
    "training_episodes": 800,
    "inference_episodes": 50,
    "minibatch_size": 64,
    "entropy_coefficient": 0.001,   # very low entropy for fine-tuning (tiny bit of randomness)
    "update_epochs": 2,             # more update epochs for fine-tuning
    "learning_rate": 1e-4,          # lower learning rate for stability near end
    "reward_shaping": {
        "key": 1.0,
        "door": 2.0,
        "room_crossing": 3,
        "ball": 4.0,
        "goal": 12.0,
        "turn_penalty": 0.0,
        "step": 0.001,
    },
})


# =====================================================================
#   SET 5: SimpleGrid — Algorithm Comparison (DQN vs A2C vs PPO)
# =====================================================================
SET5_MAX_STEPS = 200
SET5_EPISODES = 1000
SET5_INFERENCE_EPISODES = 50

SET5_REWARD_SHAPING = {
    "step": 0.005,   # light penalty to encourage efficiency
    "goal": 1.0,
}

# DQN on SimpleGrid
SET5_DQN_SG = {
    "name": "SET5_DQN_SG",
    "config": copy.deepcopy(DQN_BASE_CONFIG),
}
SET5_DQN_SG["config"].update({
    "env_name":             "SimpleGrid",
    "training_episodes":    SET5_EPISODES,
    "inference_episodes":   SET5_INFERENCE_EPISODES,
    "max_steps":            SET5_MAX_STEPS,
    "epsilon_min":          0.05,
    "target_update_freq":   1000,
    "reward_shaping":       SET5_REWARD_SHAPING,
})

# A2C on SimpleGrid
SET5_A2C_SG = {
    "name": "SET5_A2C_SG",
    "config": copy.deepcopy(A2C_BASE_CONFIG),
}
SET5_A2C_SG["config"].update({
    "env_name":             "SimpleGrid",
    "training_episodes":    SET5_EPISODES,
    "inference_episodes":   SET5_INFERENCE_EPISODES,
    "max_steps":            SET5_MAX_STEPS,
    "reward_shaping":       SET5_REWARD_SHAPING,
})

# PPO on SimpleGrid
SET5_PPO_SG = {
    "name": "SET5_PPO_SG",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
SET5_PPO_SG["config"].update({
    "env_name":             "SimpleGrid",
    "training_episodes":    SET5_EPISODES,
    "inference_episodes":   SET5_INFERENCE_EPISODES,
    "max_steps":            SET5_MAX_STEPS,
    "reward_shaping":       SET5_REWARD_SHAPING,
})

exp_set_5 = [
    SET5_DQN_SG,
    SET5_A2C_SG,
    SET5_PPO_SG,
]


# =====================================================================
#                   Smoke test configs
# =====================================================================
DQN_SIMPLEGRID_BASELINE = {"name": "smoke_DQN_SimpleGrid", "config": copy.deepcopy(DQN_BASE_CONFIG)}
A2C_SIMPLEGRID_BASELINE = {"name": "smoke_A2C_SimpleGrid", "config": copy.deepcopy(A2C_BASE_CONFIG)}
PPO_SIMPLEGRID_BASELINE = {"name": "smoke_PPO_SimpleGrid", "config": copy.deepcopy(PPO_BASE_CONFIG)}

_KDB_DQN = copy.deepcopy(DQN_BASE_CONFIG)
_KDB_DQN.update({"env_name": "KeyDoorBall", "max_steps": 450})
DQN_KEYDOORBALL_BASELINE = {"name": "smoke_DQN_KDB", "config": _KDB_DQN}

_KDB_A2C = copy.deepcopy(A2C_BASE_CONFIG)
_KDB_A2C.update({"env_name": "KeyDoorBall", "max_steps": 450})
A2C_KEYDOORBALL_BASELINE = {"name": "smoke_A2C_KDB", "config": _KDB_A2C}

_KDB_PPO = copy.deepcopy(PPO_BASE_CONFIG)
_KDB_PPO.update({"env_name": "KeyDoorBall", "max_steps": 450})
PPO_KEYDOORBALL_BASELINE = {"name": "smoke_PPO_KDB", "config": _KDB_PPO}