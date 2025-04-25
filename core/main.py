# main.py
"""
Orchestrator: sets up env, policy, risk model, and agent, then runs training.
Supports easy swapping of policy_type and risk_type via factory functions.
"""
import torch
from environment import HereditaryCancerEnv
from agent import CancerScreeningAgent
from models import make_policy_network, make_risk_model


def train_hereditary_screening(
    num_episodes: int = 500,
    n_individuals: int = 3,
    M: int = 3,
    max_time: int = 10,
    policy_type: str = 'mlp',
    risk_type: str = 'feed',
    gamma: float = 0.99,
    device: str = None
):
    """
    Main training loop for multi-cancer screening RL.
    Args:
        num_episodes: total episodes to train
        n_individuals: family size
        M: max tumor stage
        max_time: episode horizon
        policy_type: 'mlp', 'rnn', or 'transformer'
        risk_type: 'feed', 'recurrent'
        gamma: discount factor
        device: 'cpu' or 'cuda' (auto-detect if None)
    Returns:
        trained_agent (CancerScreeningAgent)
    """
    # Determine device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Build environment
    env = HereditaryCancerEnv(
        n_individuals=n_individuals,
        M=M,
        max_time=max_time,
        device=device
    )

    # Define feature dimension for policy network
    # Assuming risk model outputs this many features
    feat_dim = 32

    # Instantiate risk model and policy network via factories
    risk_model = make_risk_model(risk_type,
                                  obs_dim=env.obs_dim,
                                  feat_dim=feat_dim).to(device)
    policy_net = make_policy_network(policy_type,
                                     input_dim=feat_dim,
                                     n_individuals=n_individuals).to(device)

    # Create agent with injected modules
    agent = CancerScreeningAgent(policy=policy_net,
                                 risk_model=risk_model,
                                 gamma=gamma,
                                 device=device)

    # Training loop
    for ep in range(1, num_episodes+1):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.store_reward(reward)
        agent.update_policy()

        if ep % 50 == 0:
            print(f"Episode {ep}/{num_episodes} complete.")

    print("Training finished.")
    return agent


if __name__ == '__main__':
    trained_agent = train_hereditary_screening()
