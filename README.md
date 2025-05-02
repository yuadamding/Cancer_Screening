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
### 6. Formulation Alignment

This comprehensive guide should help you set up, run, and understand each component of the hereditary cancer screening RL pipeline. Feel free to adapt it to your research needs!

Below is a side-by-side checklist showing how our current code maps to (or deviates from) the key components of the LaTeX formulation. In a word, we correctly captured the POMDP/RL skeleton, but our prototype is a **single-cancer** version rather than the full multi-cancer model in the paper, and uses a toy HMM rather than the genotype- and covariate-driven rates in the write-up.

---

| **Paper Spec**                                                                                                                                       | **Code Implementation**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | **Status**             |
|------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| **Random genotypes** \(G_i \in\{0,1\}\) per individual, unknown initially, revealed by test or diagnosis                                       | In `environment.reset()`, we sample `self.genotypes = rng.integers(0,2,size=n)`.  We only reveal them when a positive test occurs.                                                                                                                                                                                                                                                                                                                                                                                                  | ✔️ Implemented         |
| **Hidden multi-cancer HMM** \((X_i^{(1)},…,X_i^{(C)})\) with one-way progression and stage-dependent transition rates \(\lambda_{k,k+1}^{(c)}\)      | We simulate a **single** chain per individual: `self.stages[i]` advances with fixed probability `p = 0.05*(2 if carrier else 1)`.  We do **not** support multiple cancer types (i.e.\ no vector \(X_i^{(c)}\) for \(c=1\dots C\)), nor genotype- or covariate-dependent rates as in \(\exp(\beta G + \beta_{\text{cov}}^\top Z)\).                                                                                                                                                                                         | ⚠️ Partial (single cancer, fixed rate)       |
| **Screening actions per cancer type** \(\;a_t=\{\text{which individuals, which cancer types, modality, genetic test}\}\)                            | Our `action` is a list of individual indices to screen for the **one** cancer.  No per-cancer or per-modality choice, no explicit genetic-test action.                                                                                                                                                                                                                                                                                                                                                                             | ⚠️ Partial (only “screen or not” per individual)     |
| **Stage-dependent false negatives** \(f_{\mathrm{neg}}^{(c)}(m)\)                                                                                   | We only have a constant false-negative rate of 0.2 in our simple env (`self.false_neg_prob = {m:0.2}`), not stage- or type-dependent.                                                                                                                                                                                                                                                                                                                                                                                                  | ⚠️ Simplified (constant rate)      |
| **Reward** detection reward \(r_i^{(c)}(m)-\)screening cost, summed across individuals and cancers                                                  | We use a flat `detection_reward=10` per positive test minus `screening_cost=1` per screen.  That matches the general form “detection minus cost,” but isn’t differentiated by cancer type or detection stage \(m\).                                                                                                                                                                                                                                                                                                                     | ✔️ Formally similar but toy parameters |
| **POMDP formulation** (partial observations, belief state, etc.)                                                                                    | We return partial obs tensors `[time, known_genotypes, last_test]`, use those in the agent’s `select_action`, and train with REINFORCE on the simulated POMDP.  The high-level loop (`reset`, `step`, `store_reward`, `update_policy`) follows the paper’s RL formulation.                                                                                                                                                                                                                                                            | ✔️ Fully implemented  |
| **Deep learning risk model** \(\text{NeuralNet}^{(c)}\) to map obs/history→risk features                                                               | We inject a generic `RiskModel` (e.g.\ `FeedForwardRisk`) that maps the current obs tensor to a feature vector.  We do **not** implement a separate network per cancer type or any multi-task architecture, but the factory allows swapping in more complex nets.                                                                                                                                                                                                                                                                    | ✔️ Infrastructure only (single net)      |
| **Multi-cancer RL policy** selecting per-cancer per-individual actions based on risk embeddings                                                     | Our `PolicyNetwork` produces a single Bernoulli vector of length \(N\).  To match the paper’s multi-cancer action space it would need to output an \(N\times C\) matrix of screening booleans.                                                                                                                                                                                                                                                                                                                                        | ⚠️ Partial (only single-cancer)   |
| **Baselines** “no family history” and “no Mendelian inference”                                                                                      | We added two toy baselines (random half-screen and no-screen).  These illustrate how to compare, but you could refine them (e.g.\ individual-only models using only that person’s covariates).                                                                                                                                                                                                                                                                                                                                         | ✔️ Basic baselines implemented |
| **Data generation** random pedigree + covariates + HMM trajectories + false-neg outcomes                                                             | We repurposed the original 3-generation simulator, added noise, masking, and sparsity.  That matches the spirit of “random genotypes + hidden tumor states + partial observations + noise.”                                                                                                                                                                                                                                                                                                                                          | ✔️ Implemented        |
| **Summary metrics** average return, cost, detection rate/time/stage, etc.                                                                            | We compute mean±std return for RL and baselines and save to CSV.  We do not currently break out detection time or stage per cancer type (only single type).                                                                                                                                                                                                                                                                                                                                                                        | ✔️ Partial (single-cancer)  |

---

### Conclusion & Next Steps

- **Core RL pipeline** (POMDP, REINFORCE, risk‐policy injection, baselines) **matches** the paper’s formulation.  
- **Missing pieces** for strict alignment:
  1. **Multi-cancer** support: expand `HereditaryCancerEnv` to maintain \(X_i^{(c)}\) for \(c=1\ldots C\), simulate transitions per type, and expose per-type observations.  
  2. **Genotype- and covariate-dependent HMM rates** \(\lambda_{k,k+1}^{(c)}(G_i,Z_i)\).  
  3. **Vectorized actions**: policy outputs an \(N\times C\) screening matrix, plus optional genetic test flags.  
  4. **Stage-dependent false negatives** per cancer type.  
  5. **Deep risk nets** per cancer type or multi-task architectures for \(\{R_i^{(c)}(t)\}\).  

If you’d like me to fully extend the code into a true multi-cancer POMDP (with vector states and actions), let me know and I can refactor `environment.py`, the network factories, and the agent accordingly!