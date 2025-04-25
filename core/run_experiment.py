# run_experiment.py
"""
Hereditary Cancer Screening RL Full Pipeline

Notations:
  F: # of families (--num_families)
  N: approx. # of individuals per family (--n_individuals)
  T: time horizon of each simulation (--horizon)
  M: maximum tumor stage (--max_stage)
  η: noise rate on test outcomes (--noise_level)
  φ: fraction of individuals fully unobservable (--mask_fraction)
  ψ: fraction of individuals partially observed (--sparse_fraction)
  δ: drop rate per timepoint for sparse individuals (--drop_rate)
  γ: RL discount factor (--gamma)

Steps:
 1) Generate synthetic data with masking and sparsity
 2) Instantiate POMDP environment and RL agent
 3) Define baselines:
     A) No family history (random screening)
     B) No Mendelian inference (no screening)
 4) Train RL policy on POMDP
 5) Evaluate RL and baselines
 6) Save data, models (JSON), and performance metrics

Usage:
  python run_experiment.py \
    --num_families 1000 --n_individuals 30 --horizon 200 --max_stage 3 \
    --noise_level 0.1 --mask_fraction 0.2 --sparse_fraction 0.2 --drop_rate 0.8 \
    --episodes 500 --eval_episodes 100 --policy_type mlp --risk_type feed \
    --gamma 0.99 --save_dir results --seed 42
"""
import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from core.environment import HereditaryCancerEnv
from core.agent import CancerScreeningAgent
from core.models import make_policy_network, make_risk_model
from core.simulation_data_generator import generate_data


def parse_args():
    parser = argparse.ArgumentParser(description="Hereditary Cancer Screening RL")
    # Dataset parameters
    parser.add_argument('--num_families', type=int, default=1000,
                        help='F: number of synthetic families')
    parser.add_argument('--n_individuals', type=int, default=30,
                        help='N: approx size per family')
    parser.add_argument('--horizon', type=int, default=200,
                        help='T: time horizon')
    parser.add_argument('--max_stage', type=int, default=3,
                        help='M: max tumor stage')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='η: probability to flip test outcome')
    parser.add_argument('--mask_fraction', type=float, default=0.2,
                        help='φ: fraction fully unobservable members')
    parser.add_argument('--sparse_fraction', type=float, default=0.2,
                        help='ψ: fraction partially observed members')
    parser.add_argument('--drop_rate', type=float, default=0.8,
                        help='δ: drop rate for sparse members')
    # RL parameters
    parser.add_argument('--episodes', type=int, default=500,
                        help='RL training episodes')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Evaluation episodes')
    parser.add_argument('--policy_type', type=str, choices=['mlp','rnn','transformer'],
                        default='mlp', help='Policy network type')
    parser.add_argument('--risk_type', type=str, choices=['feed','rnn'],
                        default='feed', help='Risk model type')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='γ: discount factor')
    # Misc
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu or cuda')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Output folder')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args, _ = parser.parse_known_args()
    return args


def generate_masked_data(args):
    """
    Generate F families with noise η, full-mask φ, sparsity ψ, drop δ.
    Returns combined DataFrame.
    """
    rng = np.random.default_rng(args.seed)
    buffer = []
    for fid in range(args.num_families):
        df = generate_data(family_id=fid, T=args.horizon, M=args.max_stage,
                           seed=args.seed + fid)
        # unify column
        if 'test' in df: df.rename(columns={'test':'test_outcome'}, inplace=True)
        # add noise η
        flip = rng.random(len(df)) < args.noise_level
        df.loc[flip, 'test_outcome'] = 1 - df.loc[flip, 'test_outcome']
        # full mask φ
        ids = df['individual_id'].unique()
        n_full = max(1, int(len(ids)*args.mask_fraction))
        full = rng.choice(ids, n_full, replace=False)
        df.loc[df['individual_id'].isin(full), ['genotype','test_outcome']] = np.nan
        # sparse ψ
        rem = [i for i in ids if i not in full]
        n_sparse = max(1, int(len(ids)*args.sparse_fraction))
        sparse = rng.choice(rem, n_sparse, replace=False)
        for sid in sparse:
            idx = df.index[df['individual_id']==sid]
            drop_idx = rng.random(len(idx)) < args.drop_rate
            df.loc[idx[drop_idx], 'test_outcome'] = np.nan
        buffer.append(df)
    return pd.concat(buffer, ignore_index=True)


def train_agent(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    env = HereditaryCancerEnv(n_individuals=args.n_individuals,
                               M=args.max_stage,
                               max_time=args.horizon,
                               device=device)
    feat_dim = 32
    risk = make_risk_model(args.risk_type, obs_dim=env.obs_dim,
                           feat_dim=feat_dim).to(device)
    policy = make_policy_network(args.policy_type,
                                 input_dim=feat_dim,
                                 n_individuals=args.n_individuals).to(device)
    agent = CancerScreeningAgent(policy, risk, gamma=args.gamma,
                                 device=device)
    for ep in range(1, args.episodes+1):
        obs = env.reset(); done=False
        while not done:
            act = agent.select_action(obs)
            obs, r, done, _ = env.step(act)
            agent.store_reward(r)
        agent.update_policy()
        if ep % 100 == 0:
            print(f"Trained {ep}/{args.episodes} episodes")
    return agent


def baseline_random(env, args):
    """Baseline A: ignore family history, random half screening."""
    res = []
    for _ in range(args.eval_episodes):
        obs, done, total = env.reset(), False, 0
        while not done:
            k = env.n_individuals//2
            act = list(np.random.choice(env.n_individuals, k, False))
            obs, r, done, _ = env.step(act); total += r
        res.append(total)
    return np.mean(res), np.std(res)


def baseline_none(env, args):
    """Baseline B: no Mendelian inference, never screen."""
    res = []
    for _ in range(args.eval_episodes):
        obs, done, total = env.reset(), False, 0
        while not done:
            obs, r, done, _ = env.step([]); total += r
        res.append(total)
    return np.mean(res), np.std(res)


def evaluate_all(agent, args):
    # RL
    env_rl = HereditaryCancerEnv(args.n_individuals, args.max_stage,
                                  args.horizon, device=agent.device)
    rl_vals = [run_episode(env_rl, agent) for _ in range(args.eval_episodes)]
    rl_mean, rl_std = np.mean(rl_vals), np.std(rl_vals)
    # Baselines
    env_b1 = HereditaryCancerEnv(args.n_individuals, args.max_stage,
                                 args.horizon, device=agent.device)
    b1_mean, b1_std = baseline_random(env_b1, args)
    env_b2 = HereditaryCancerEnv(args.n_individuals, args.max_stage,
                                 args.horizon, device=agent.device)
    b2_mean, b2_std = baseline_none(env_b2, args)
    # Compile
    df = pd.DataFrame({
        'method': ['RL','RandomHalf','NoScreen'],
        'mean_return': [rl_mean, b1_mean, b2_mean],
        'std_return': [rl_std, b1_std, b2_std]
    })
    return df


def run_episode(env, agent):
    obs, done, total = env.reset(), False, 0
    while not done:
        a = agent.select_action(obs)
        obs, r, done, _ = env.step(a)
        total += r
    return total


def save_models(agent, args):
    os.makedirs(args.save_dir, exist_ok=True)
    for name, mdl in [('policy',agent.policy),('risk',agent.risk_model)]:
        sd = mdl.state_dict(); jd = {k:v.cpu().numpy().tolist() for k,v in sd.items()}
        with open(os.path.join(args.save_dir,f"{name}.json"),'w') as f:
            json.dump(jd, f)



def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    # 1) Data
    print(f"Generating data F={args.num_families}, N≈{args.n_individuals}, T={args.horizon}")
    df = generate_masked_data(args)
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(os.path.join(args.save_dir,'synthetic_data.csv'), index=False)
    # 2) Train
    agent = train_agent(args)
    # 3) Evaluate
    results_df = evaluate_all(agent, args)
    results_df.to_csv(os.path.join(args.save_dir,'comparison_results.csv'), index=False)
    print(results_df)
    # 4) Save
    save_models(agent, args)
    print("Finished pipeline. Outputs in", args.save_dir)

if __name__=='__main__':
    main()