# run_experiment.py
"""
Top-level script to:
  1) Generate synthetic family data for multi-cancer risk modeling
  2) Instantiate environment, models, and RL agent
  3) Train the RL agent for hereditary cancer screening
  4) Evaluate performance with comprehensive metrics
  5) Save trained models (policy & risk) and synthetic data
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
    parser = argparse.ArgumentParser("Hereditary Cancer Screening RL")
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--n_individuals', type=int, default=3,
                        help='Family size')
    parser.add_argument('--max_stage', type=int, default=3,
                        help='Maximum tumor stage (M)')
    parser.add_argument('--horizon', type=int, default=10,
                        help='Time steps per episode')
    parser.add_argument('--policy_type', type=str, choices=['mlp','rnn','transformer'],
                        default='mlp', help='Policy network architecture')
    parser.add_argument('--risk_type', type=str, choices=['feed','rnn'],
                        default='feed', help='Risk model architecture')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--device', type=str, default=None,
                        help='cpu or cuda (auto if not set)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save trained models and data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data generation and simulation')
    args, _ = parser.parse_known_args()
    return args


def train_agent(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    env = HereditaryCancerEnv(
        n_individuals=args.n_individuals,
        M=args.max_stage,
        max_time=args.horizon,
        device=device
    )
    feat_dim = 32
    risk_model = make_risk_model(
        args.risk_type,
        obs_dim=env.obs_dim,
        feat_dim=feat_dim
    ).to(device)
    policy_net = make_policy_network(
        args.policy_type,
        input_dim=feat_dim,
        n_individuals=args.n_individuals
    ).to(device)
    agent = CancerScreeningAgent(
        policy=policy_net,
        risk_model=risk_model,
        gamma=args.gamma,
        device=device
    )
    for ep in range(1, args.episodes+1):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.store_reward(reward)
        agent.update_policy()
        if ep % 50 == 0:
            print(f"[Train] Episode {ep}/{args.episodes} complete.")
    return agent


def evaluate_agent(agent, args):
    env = HereditaryCancerEnv(
        n_individuals=args.n_individuals,
        M=args.max_stage,
        max_time=args.horizon,
        device=agent.device
    )
    total_returns = []
    detection_times = []
    detection_stages = []
    cumulative_costs = []
    for _ in range(args.eval_episodes):
        obs = env.reset()
        done = False
        t = 0
        episode_return = 0.0
        detected = [False]*args.n_individuals
        detect_time = [args.horizon+1]*args.n_individuals
        detect_stage = [None]*args.n_individuals
        cost = 0.0
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            cost += len(action) * env.screening_cost
            for idx in action:
                if not detected[idx] and env.last_tests[idx] == 1:
                    detected[idx] = True
                    detect_time[idx] = t
                    detect_stage[idx] = env.stages[idx]
            t += 1
        total_returns.append(episode_return)
        cumulative_costs.append(cost)
        for i in range(args.n_individuals):
            if detected[i]:
                detection_times.append(detect_time[i])
                detection_stages.append(detect_stage[i])
    avg_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    avg_cost = np.mean(cumulative_costs)
    det_rate = len(detection_times)/(args.eval_episodes*args.n_individuals)
    avg_det_time = np.mean(detection_times) if detection_times else np.nan
    avg_det_stage = np.mean(detection_stages) if detection_stages else np.nan
    print(f"[Eval] Return: {avg_return:.2f} ± {std_return:.2f}")
    print(f"[Eval] Avg cost: {avg_cost:.2f}")
    print(f"[Eval] Detection rate: {det_rate*100:.1f}%")
    print(f"[Eval] Avg detection time: {avg_det_time:.2f}")
    print(f"[Eval] Avg detection stage: {avg_det_stage:.2f}")
    return {
        'avg_return': avg_return,
        'std_return': std_return,
        'avg_cost': avg_cost,
        'detection_rate': det_rate,
        'avg_detection_time': avg_det_time,
        'avg_detection_stage': avg_det_stage
    }


def save_models(agent, args):
    os.makedirs(args.save_dir, exist_ok=True)
    # Export policy state dict to JSON
    policy_sd = agent.policy.state_dict()
    policy_json = {k: policy_sd[k].cpu().numpy().tolist() for k in policy_sd}
    with open(os.path.join(args.save_dir, 'policy.json'), 'w') as f:
        json.dump(policy_json, f)
    # Export risk model state dict to JSON
    risk_sd = agent.risk_model.state_dict()
    risk_json = {k: risk_sd[k].cpu().numpy().tolist() for k in risk_sd}
    with open(os.path.join(args.save_dir, 'risk_model.json'), 'w') as f:
        json.dump(risk_json, f)
    print(f"Models saved in JSON to {args.save_dir}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Generating synthetic data...")
    df = generate_data(
        family_id=0,
        T=args.horizon,
        M=args.max_stage,
        seed=args.seed
    )
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(os.path.join(args.save_dir, 'synthetic_data.csv'), index=False)
    print("Synthetic data saved.")
    agent = train_agent(args)
    metrics = evaluate_agent(agent, args)
    save_models(agent, args)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.save_dir, 'performance_metrics.csv'), index=False)
    print("Metrics saved.")

if __name__ == '__main__':
    main()
