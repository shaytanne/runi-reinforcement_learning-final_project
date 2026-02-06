import time

# base config to be extended/overridden by experiment configs
BASE_CONFIG = {
    "seed": int(time.time()),
    "training_episodes": 5,
    "inference_episodes": 10,
    "obs_shape": (84, 84, 1),
    "training_freq": 1,          # train every N steps   
}

# DQN on SimpleGrid
DEFAULT_DQN_CONFIG = BASE_CONFIG.copy()
DEFAULT_DQN_CONFIG.update({
    "algo": "DQN",
    "gamma": 0.99,               
    "batch_size": 32,            
    "buffer_capacity": 100000,   # replay buffer for DQN 
    "learning_rate": 2.5e-4,     
    "epsilon_start": 1.0,        # initial epsilon
    "epsilon_min": 0.05,         # minimum epsilon
    "epsilon_decay": 0.995,
    "training_freq": 4,          # train (call _learn) every N steps   
    "target_update_freq": 1000,  # steps between syncing target network
    "grad_clip": 1.0,            # gradient clipping value
})


# DQN on KeyDoorBall
KEY_DOOR_CONFIG = DEFAULT_DQN_CONFIG.copy()
KEY_DOOR_CONFIG.update({
    "buffer_capacity": 200000,       # more memory for longer episodes
    "epsilon_decay": 0.998,      # slower decay to explore longer
})