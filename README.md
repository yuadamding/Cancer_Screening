## Hereditary Cancer Screening RL: Code Overview & Instructions

This document guides you through the entire codebase—from basic setup and dependencies, to high-level architecture, and detailed usage instructions.

---

### 1. Basic Setup

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```

2. **Create a Python environment** (e.g., using `conda` or `venv`):
   ```bash
   conda create -n cancer-rl python=3.9
   conda activate cancer-rl
   ```

3. **Install dependencies**:
   ```bash
   pip install torch numpy pandas
   ```
   - **PyTorch**: For neural networks and RL agent.
   - **NumPy** & **Pandas**: For data simulation & metrics.

---

### 2. Directory Structure

```
project_root/
├── core/
│   ├── environment.py          # POMDP environment class
│   ├── agent.py                # RL agent implementation (REINFORCE)
│   ├── models.py               # Policy & risk model definitions + factories
│   └── simulation_data_generator.py  # Synthetic family data pipeline
├── run_experiment.py           # Orchestrator: data, train, evaluate, save
└── results/                    # Output folder (data, models, metrics)
```

- **core/environment.py**: Implements `HereditaryCancerEnv`, a POMDP with:
  - Uncertain genotypes, hidden tumor stages (0…M).
  - Screening tests with stage-dependent false-negatives.
  - Costs/rewards per screening & detection.
  - Methods: `reset()`, `step(action)`, and `_get_obs()` → tensor obs.

- **core/agent.py**: Defines `CancerScreeningAgent`:
  - Injected **policy** and **risk** networks.
  - **select_action(obs)** → uses risk model to extract features, policy to sample Bernoulli actions.
  - **store_reward(r)** and **update_policy()** implement REINFORCE.

- **core/models.py**: Contains abstract & concrete classes:
  - **PolicyNetwork**s: `FeedForwardPolicy`, `RecurrentPolicy`, `TransformerPolicy`.
  - **RiskModel**s: `FeedForwardRisk`, `RecurrentRisk`.
  - Factories: `make_policy_network(type, ...)`, `make_risk_model(type, ...)`.

- **core/simulation_data_generator.py**: Generates synthetic pedigree data:
  - Builds 3-generation families (~30–40 members) via Mendelian inheritance.
  - Assigns covariates (smoking, exercise, etc.).
  - Simulates multi-stage HMM tumor trajectories.
  - Produces false-negative test outcomes.
  - **Function**: `generate_data(family_id, T, M, ...)` → `pandas.DataFrame`.

- **run_experiment.py**: End-to-end pipeline (see next section).

---

### 3. High-Level Workflow in `run_experiment.py`

1. **Parse arguments** (`argparse`):
   - Data params: `--num_families F`, `--n_individuals N`, `--horizon T`, `--max_stage M`,
     `--noise_level η`, `--mask_fraction φ`, `--sparse_fraction ψ`, `--drop_rate δ`, `--seed`.
   - RL params: `--episodes`, `--eval_episodes`, `--policy_type`, `--risk_type`, `--gamma γ`.
   - I/O: `--save_dir`, `--device`.

2. **Data Generation**:
   ```python
   df = generate_masked_data(args)
   df.to_csv('results/synthetic_data.csv')
   ```
   - **Masks** φ of individuals entirely (no genotype/test).
   - **Sparsifies** ψ of individuals by dropping δ of their timepoints.
   - **Injects noise** η by flipping test outcomes.

3. **Environment & Agent Setup**:
   ```python
   env = HereditaryCancerEnv(N, M, T, device)
   risk = make_risk_model(risk_type, obs_dim=env.obs_dim, feat_dim)
   policy = make_policy_network(policy_type, feat_dim, N)
   agent = CancerScreeningAgent(policy, risk, gamma, device)
   ```

4. **RL Training**:
   - Loop for `episodes`:
     - `obs = env.reset()`
     - While not done: choose action, step env, store rewards
     - After episode: `agent.update_policy()`

5. **Baselines**:
   - **RandomHalf**: randomly screens half of individuals each step.
   - **NoScreen**: never screens (empty action).

6. **Evaluation**:
   - Run `eval_episodes` for RL and each baseline.
   - Collect total return per episode → compute mean ± std.
   - Compile into a `pandas.DataFrame` and save as `comparison_results.csv`.

7. **Save Models**:
   - Export `policy` and `risk_model` weights as JSON (`.json` files).
   - Save performance metrics as CSV.

---

### 4. Running the Code

```bash
# Basic example:
python run_experiment.py \
  --num_families 1000 --n_individuals 30 --horizon 200 \
  --noise_level 0.1 --mask_fraction 0.2 --sparse_fraction 0.2 --drop_rate 0.8 \
  --episodes 500 --eval_episodes 100 \
  --policy_type mlp --risk_type feed --gamma 0.99 \
  --save_dir results --seed 42
```

- **Outputs** in `results/`:
  - `synthetic_data.csv`: F×N×T rows of simulated data.
  - `policy.json`, `risk_model.json`: model weights.
  - `comparison_results.csv`: performance (mean ± std) of RL vs baselines.

---

### 5. Extending & Customizing

- **New Baselines**: implement additional `baseline_*` functions.
- **Model Variants**: add new `PolicyNetwork` or `RiskModel` in `core/models.py`, register in factories.
- **Data Complexity**: adjust masking/sparsity parameters or HMM rates in `simulation_data_generator.py`.

---

This comprehensive guide should help you set up, run, and understand each component of the hereditary cancer screening RL pipeline. Feel free to adapt it to your research needs!

