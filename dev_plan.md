# Dev Plan

## Phase 1: Infra + Setup - ✅ DONE
**Goal:** Establish a robust environment for developing, testing, and visualizing RL agents without crashing local machines.

- [x] **Project Structure:** Created `src/` modules (`agent`, `buffer`, `model`, `trainer`, `utils`, `template`) and `main.py`.
- [x] **"Twin Template" Workflow:**
  - `src/local_template.py`: Modified for safe local development (no pop-ups).
  - `tests/official_template.py`: Pristine copy for final compatibility verification.
- [x] **Logging & Visualization:**
  - `Logger` (CSV): Tracks Reward, Steps, Epsilon, and Success Rate.
  - `plot_training_curves`: Generates smoothed graphs for Rewards, Steps, and Success.
  - `VideoRecorder`: Captures "Middle" (Training) and "Final" (Inference) agent behaviors.
- [x] **Verification:**
  - Implemented `RandomAgent` to test the pipeline.
  - Verified that `train()` and `evaluate()` loops produce correct logs and videos.

---

## Phase 2: Core Agent (DQL)
**Goal:** Solve `SimpleGridEnv` using a standard DQN implementation.

### 2.1 Preprocessing 
- [x] **Update `pre_process`:** Resize images to **84x84** (Grayscale) to reduce memory usage.
- [x] **Update Config:** Ensure `obs_shape` in `SimpleGridEnv` and `main` config matches `(84, 84, 1)`.

### 2.2 The Brain (Model)
- [ ] **Implement `MiniGridCNN` (`src/model.py`):**
  - Input: `(Batch, 1, 84, 84)`
  - Architecture: 3 Conv layers (ReLU) + Flatten + Fully Connected layers.
  - Output: Q-values for each action (Size `num_actions`).

### 2.3 The Memory (Buffer)
- [ ] **Implement `ReplayBuffer` (`src/buffer.py`):**
  - Storage: Efficient `numpy` arrays (State, Action, Reward, Next_State, Done).
  - Data Type: Store States as `uint8` (0-255) to save RAM.
  - Methods: `add()`, `sample()`, `__len__()`.

### 2.4 The Algorithm (Agent)
- [ ] **Implement `DQNAgent` (`src/agent.py`):**
  - **Init:** Initialize Policy Net, Target Net, Optimizer (Adam).
  - **Act:** Epsilon-Greedy logic.
  - **Step:** Store transition in Buffer -> Sample Batch -> Compute Loss (MSE/Huber) -> Backprop -> Update Target Network (periodically).
  - **Normalization:** Convert `uint8` batch to `float32` and divide by 255.0 *inside* the training step.

### 2.5 Train & Debug
- [ ] **Run Training:** Train on `SimpleGridEnv`.
- [ ] **Verify Success:** Watch the "Success Rate" plot rise from 0% to ~100%.

---

## Phase 3: Advanced Algorithms (The "Competition")
**Goal:** Improve performance and stability to meet the "compare different approaches" requirement.

- [ ] **Double DQN:**
  - Modify `agent.py` to decouple action selection from Q-value estimation in the target calculation.
- [ ] **Comparison:**
  - Run both DQN and Double DQN.
  - Use `plot_comparison()` (`src/utils.py`) to overlay their learning curves.

---

## Phase 4: The Complex Environment (`KeyDoorBallEnv`)
**Goal:** Solve the multi-room environment with sparse rewards.

- [ ] **Transfer Agent:** Run the working Double DQN on `KeyDoorBallEnv`.
- [ ] **Reward Shaping (`src/template.py`):**
  - Modify `step()` to provide intermediate rewards (e.g., +0.1 for picking up Key, +0.2 for opening Door).
- [ ] **Hyperparameter Tuning:**
  - Increase `buffer_size` and `epsilon_decay` for the harder task.

---

## Phase 5: Submission & Reporting
**Goal:** Package everything for the teaching staff.

- [ ] **Final Verification:** Run `tests/verify.py` to ensure custom classes work with the official template.
- [ ] **Notebook Assembly:**
  - Copy `pre_process`, `MiniGridCNN`, `ReplayBuffer`, and `DQNAgent` into `tests/official_template.py`.
  - Run the notebook on Google Colab.
- [ ] **Report Generation:**
  - Use stats from `inference_report.txt` (Avg Steps, Success Rate).
  - Include `training_curves.png` and `comparison_plot.png`.
  - Embed the "Middle" and "Final" videos.