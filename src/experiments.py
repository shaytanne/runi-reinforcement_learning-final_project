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

PPO_BASE_CONFIG = copy.deepcopy(A2C_BASE_CONFIG)   # inherits from A2C, NOT SHARED
PPO_BASE_CONFIG.update({
    "algo": "PPO",

    # PPO-specific
    "entropy_coefficient": 0.01,
    "clip_eps": 0.2,
    "update_epochs": 4,
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

SET4_PPO_FINE_TUNE_GREEDY2 = {
    "name": "SET4_PPO_Fine_Tune_Greedy2",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
SET4_PPO_FINE_TUNE_GREEDY2["config"].update({
    "env_name": "KeyDoorBall",
    "obs_shape": (84, 84, 1),
    "max_steps": 450,
    "seed": 1772571932,

    # shorter + cheaper than full training
    "training_episodes": 600,
    "inference_episodes": 50,

    # PPO stability / determinization
    "minibatch_size": 64,
    "update_epochs": 2,
    "learning_rate": 5e-5,          # smaller than 1e-4 to avoid drifting
    "entropy_coefficient": 0.0,     # key change: force determinization

    # keep what worked
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "max_grad_norm": 0.5,
    "value_loss_coefficient": 0.5,

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

# simply adds more episodes to base training on a pretrained agent
SET4_PPO_EXTEND = {
    "name": "SET4_PPO_Extend",
    "config": copy.deepcopy(SET4_PPO_KDB["config"]),  
}
SET4_PPO_EXTEND["config"].update({
    "training_episodes": 1000,   # train another 1000 episodes from checkpoint
    "inference_episodes": 100,   # for report
})

SET4_PPO_FINE_TUNE_POLISH = {
    "name": "SET4_PPO_FineTune_Polish",
    "config": copy.deepcopy(SET4_PPO_KDB["config"]),  
}
SET4_PPO_FINE_TUNE_POLISH["config"].update({
    "training_episodes": 400,     # start with 400, extend in 200-chunks if needed
    "inference_episodes": 100,    # for report

    # adapted params for fine-tuning phase
    "learning_rate": 1e-4,
    "update_epochs": 2,
    "entropy_coefficient": 0.001,
})


# =====================================================================
#   SET 6: no improvement methods PPO playground
# =====================================================================

SET6_SAFE_BASE = {
    "env_name": "KeyDoorBall",
    "obs_shape": (84, 84, 1),
    "max_steps": 450,
    "seed": 1772571932,

    "training_episodes": 3000,
    "inference_episodes": 100,

    "minibatch_size": 64,
    "clip_eps": 0.2,
    "max_grad_norm": 0.5,
    "value_loss_coefficient": 0.5,

    "reward_shaping": {
        "key": 1.0,
        "door": 2.0,
        "room_crossing": 3.0,
        "ball": 4.0,
        "goal": 12.0,
        "turn_penalty": 0.0,
        "step": 0.001,
    },
}

# reward-to-go returns + lower LR + less epochs + LOW entropy
EXP6A_PPO_SAFE_RTG_LR1E4_E2_ENT1E3 = {
    "name": "EXP6A_PPO_SAFE_RTG_LR1E4_E2_ENT1E3",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
EXP6A_PPO_SAFE_RTG_LR1E4_E2_ENT1E3["config"].update(SET6_SAFE_BASE)
EXP6A_PPO_SAFE_RTG_LR1E4_E2_ENT1E3["config"].update({
    "return_mode": "reward_to_go",
    "learning_rate": 1e-4,
    "update_epochs": 2,
    "entropy_coefficient": 0.001,
})

# same as 6A, a bit more exploration
EXP6B_PPO_SAFE_RTG_LR1E4_E2_ENT5E3 = {
    "name": "EXP6B_PPO_SAFE_RTG_LR1E4_E2_ENT5E3",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
EXP6B_PPO_SAFE_RTG_LR1E4_E2_ENT5E3["config"].update(SET6_SAFE_BASE)
EXP6B_PPO_SAFE_RTG_LR1E4_E2_ENT5E3["config"].update({
    "return_mode": "reward_to_go",
    "learning_rate": 1e-4,
    "update_epochs": 2,
    "entropy_coefficient": 0.005,
})

# same as 6A, more optimization per rollout
EXP6C_PPO_SAFE_RTG_LR1E4_E4_ENT1E3 = {
    "name": "EXP6C_PPO_SAFE_RTG_LR1E4_E4_ENT1E3",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
EXP6C_PPO_SAFE_RTG_LR1E4_E4_ENT1E3["config"].update(SET6_SAFE_BASE)
EXP6C_PPO_SAFE_RTG_LR1E4_E4_ENT1E3["config"].update({
    "return_mode": "reward_to_go",
    "learning_rate": 1e-4,
    "update_epochs": 4,
    "entropy_coefficient": 0.001,
})

# same as 6A, with old higher LR
EXP6D_PPO_SAFE_RTG_LR3E4_E2_ENT1E3 = {
    "name": "EXP6D_PPO_SAFE_RTG_LR3E4_E2_ENT1E3",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
EXP6D_PPO_SAFE_RTG_LR3E4_E2_ENT1E3["config"].update(SET6_SAFE_BASE)
EXP6D_PPO_SAFE_RTG_LR3E4_E2_ENT1E3["config"].update({
    "return_mode": "reward_to_go",
    "learning_rate": 3e-4,
    "update_epochs": 2,
    "entropy_coefficient": 0.001,
})

# control run: td(0) estimator, gentler hyperparams than first basic PPO exp
EXP6E_PPO_SAFE_TD0_LR1E4_E2_ENT1E3 = {
    "name": "EXP6E_PPO_SAFE_TD0_LR1E4_E2_ENT1E3",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
EXP6E_PPO_SAFE_TD0_LR1E4_E2_ENT1E3["config"].update(SET6_SAFE_BASE)
EXP6E_PPO_SAFE_TD0_LR1E4_E2_ENT1E3["config"].update({
    "return_mode": "td0",
    "learning_rate": 1e-4,
    "update_epochs": 2,
    "entropy_coefficient": 0.001,
})

exp_set_6 = [
    EXP6A_PPO_SAFE_RTG_LR1E4_E2_ENT1E3,
    EXP6B_PPO_SAFE_RTG_LR1E4_E2_ENT5E3,
    EXP6C_PPO_SAFE_RTG_LR1E4_E4_ENT1E3,
    EXP6D_PPO_SAFE_RTG_LR3E4_E2_ENT1E3,
    EXP6E_PPO_SAFE_TD0_LR1E4_E2_ENT1E3,
]


# =====================================================================
#   SET 7: PPO + GAE
# =====================================================================
SET7_BASE_CONFIG = {
    "env_name": "KeyDoorBall",
    "obs_shape": (84, 84, 1),
    "max_steps": 450,
    "seed": 1772571932,

    "training_episodes": 3000,
    "inference_episodes": 100,

    "gamma": 0.99,
    "minibatch_size": 64,

    "reward_shaping": {
        "key": 1.0,
        "door": 2.0,
        "room_crossing": 3.0,
        "ball": 4.0,
        "goal": 12.0,
        "turn_penalty": 0.0,
        "step": 0.001,
    },
}

EXP7A_PPO_GAE_ONLY = {
    "name": "EXP7A_PPO_GAE_ONLY",
    "config": copy.deepcopy(PPO_BASE_CONFIG),
}
EXP7A_PPO_GAE_ONLY["config"].update(SET7_BASE_CONFIG)
EXP7A_PPO_GAE_ONLY["config"].update({
    "learning_rate": 1e-4,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "update_epochs": 2,

    "value_loss_coefficient": 0.5,
    "entropy_coefficient": 0.001,
    "max_grad_norm": 0.5,
})

EXP7A_PPO_GAE_EXTEND = {
    "name": "EXP7A_PPO_GAE_EXTEND",
    "config": copy.deepcopy(EXP7A_PPO_GAE_ONLY["config"]),
}
EXP7A_PPO_GAE_EXTEND["config"].update({
    "training_episodes": 1000,   # additional episodes after loading 7A checkpoint
    "inference_episodes": 100,
})

EXP7A_PPO_GAE_FINE_TUNE = {
    "name": "EXP7A_PPO_GAE_FINE_TUNE",
    "config": copy.deepcopy(EXP7A_PPO_GAE_ONLY["config"]),
}
EXP7A_PPO_GAE_FINE_TUNE["config"].update({
    "training_episodes": 400,
    "inference_episodes": 100,

    "learning_rate": 5e-5,
    "update_epochs": 2,
    "entropy_coefficient": 5e-4,

    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "minibatch_size": 64,
    "max_grad_norm": 0.5,
    "value_loss_coefficient": 0.5,
})

EXP7B_PPO_GAE_ONLY_FAILED_INTERACTION = {
    "name": "EXP7B_PPO_GAE_ONLY_FAILED_INTERACTION",
    "config": copy.deepcopy(EXP7A_PPO_GAE_ONLY["config"]),
}
EXP7B_PPO_GAE_ONLY_FAILED_INTERACTION["config"].update({
    "reward_shaping": {
        "key": 1.0,
        "door": 2.0,
        "room_crossing": 3.0,
        "ball": 4.0,
        "goal": 12.0,
        "step": 0.001,
        "failed_pickup_penalty": 0.003,
        "failed_toggle_penalty": 0.003,
    },
})

EXP7C_PPO_GAE_RGB = {
    "name": "EXP7C_PPO_GAE_RGB",
    "config": copy.deepcopy(EXP7A_PPO_GAE_ONLY["config"]),
}
EXP7C_PPO_GAE_RGB["config"].update({
    "obs_shape": (84, 84, 3),
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