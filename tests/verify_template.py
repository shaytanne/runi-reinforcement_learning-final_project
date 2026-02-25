"""
Verifies src/template.py is aligned/compatible with OFFICIAL (unmodified) template
- pre_process
- envs
- agents

Run with: pytest tests/verify_template.py -v
         or standalone: python tests/verify_template.py
"""

import sys
import os
import numpy as np
import torch

# enable importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import the OFFICIAL (clean) template
# might print stuff/open windows
from tests.official_template import SimpleGridEnv as OfficialSimpleGridEnv
from tests.official_template import KeyDoorBallEnv as OfficialKeyDoorBallEnv

# import LOCAL implementations
from src.template import pre_process 
from src.agent import DQNAgent, A2CAgent
from src.experiments import PROJECT_BASE_CONFIG

# --- preprocessing ---

def test_preprocessing_simplegrid_shape_and_dtype():
    """
    pre_process injected into the OFFICIAL SimpleGridEnv must return
    (84, 84, 1) uint8 (expected by all agents)
    """
    env = OfficialSimpleGridEnv(preprocess=pre_process, max_steps=10)
    obs, _ = env.reset()

    assert obs.shape == (84, 84, 1), (
        f"Shape mismatch: got {obs.shape}, expected (84, 84, 1)"
    )
    assert obs.dtype == np.uint8, (
        f"Dtype mismatch: got {obs.dtype}, expected uint8"
    )

def test_preprocessing_keydoorball_shape_and_dtype():
    """
    pre_process injected into the OFFICIAL KeyDoorBallEnv must return
    (84, 84, 1) uint8 (expected by all agents)
    """
    env = OfficialKeyDoorBallEnv(preprocess=pre_process, max_steps=10)
    obs, _ = env.reset()

    assert obs.shape == (84, 84, 1), (
        f"Shape mismatch: got {obs.shape}, expected (84, 84, 1)"
    )
    assert obs.dtype == np.uint8, (
        f"Dtype mismatch: got {obs.dtype}, expected uint8"
    )

def test_preprocessing_step_consistency():
    """Observations returned by step() and reset() must have same shape/dtype"""
    env = OfficialSimpleGridEnv(preprocess=pre_process, max_steps=10)
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, _, _, _, _ = env.step(action)

    assert next_obs.shape == obs.shape, (
        f"step() shape {next_obs.shape} != reset() shape {obs.shape}"
    )
    assert next_obs.dtype == obs.dtype, (
        f"step() dtype {next_obs.dtype} != reset() dtype {obs.dtype}"
    )


# --- agents ---

def test_dqn_agent_action_on_official_env():
    """
    DQNAgent.choose_action() must accept observations from the OFFICIAL env
    and return valid action index
    """
    env = OfficialSimpleGridEnv(preprocess=pre_process, max_steps=10)
    obs, _ = env.reset()
    num_actions = env.action_space.n

    agent = DQNAgent(
        config=PROJECT_BASE_CONFIG,
        obs_shape=(84, 84, 1),
        num_actions=num_actions,
        device=torch.device("cpu")
    )

    action = agent.choose_action(obs)
    assert isinstance(action, int), f"Expected int, got {type(action)}"
    assert 0 <= action < num_actions, f"Action {action} out of range [0, {num_actions})"

def test_a2c_agent_action_on_official_env():
    """
    A2CAgent.choose_action() must accept observations from the OFFICIAL env
    and return valid action index
    """
    env = OfficialSimpleGridEnv(preprocess=pre_process, max_steps=10)
    obs, _ = env.reset()
    num_actions = env.action_space.n

    agent = A2CAgent(
        config=PROJECT_BASE_CONFIG,
        obs_shape=(84, 84, 1),
        num_actions=num_actions,
        device=torch.device("cpu")
    )

    action = agent.choose_action(obs)
    assert isinstance(action, int), f"Expected int, got {type(action)}"
    assert 0 <= action < num_actions, f"Action {action} out of range [0, {num_actions})"


# --- main block (to run as standalone) ---
def run_verification():
    print("\nStarting verification against official template...")
    tests = [
        test_preprocessing_simplegrid_shape_and_dtype,
        test_preprocessing_keydoorball_shape_and_dtype,
        test_preprocessing_step_consistency,
        test_dqn_agent_action_on_official_env,
        test_a2c_agent_action_on_official_env,
    ]
    for test in tests:
        try:
            test()
            print(f"  {test.__name__}")
        except Exception as e:
            print(f"  {test.__name__}: {e}")
    print("\nVerification complete.\n")


if __name__ == "__main__":
    run_verification()
