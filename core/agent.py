"""
CancerScreeningAgent with injected policy and risk models.
Maintains high cohesion (focus on RL logic) and low coupling (policy and risk models are external).
"""
import torch

class CancerScreeningAgent:
    def __init__(self, policy, risk_model, gamma=0.99, device='cuda'):
        """
        Args:
            policy (nn.Module): maps feature vectors to logits of screening decisions
            risk_model (nn.Module): maps raw observations to feature vectors
            gamma (float): discount factor for returns
            device (str): 'cpu' or 'cuda'
        """
        self.policy = policy.to(device)
        self.risk_model = risk_model.to(device)
        self.gamma = gamma
        self.device = device
        # optimize both policy and risk_model parameters
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.risk_model.parameters()),
            lr=1e-3
        )
        # buffers for policy gradient
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        """
        Given a batched observation tensor obs: [batch_size, obs_dim],
        1) compute feature embeddings via risk_model
        2) get logits from policy
        3) sample Bernoulli actions per individual
        Returns the list of selected individual indices for screening.
        """
        # Move to correct device
        obs = obs.to(self.device)
        # Extract features
        features = self.risk_model(obs)               # [batch, feat_dim]
        # Policy produces logits per individual
        logits = self.policy(features)               # [batch, n_individuals]
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        # Sample screening decisions
        dist = torch.distributions.Bernoulli(probs)
        sample = dist.sample()                      # [batch, n_individuals]
        # Store log-probabilities for training
        logp = dist.log_prob(sample).sum(dim=1)     # [batch]
        self.log_probs.append(logp)
        # Convert to list of indices (batch_size assumed=1)
        sel = sample.bool().nonzero(as_tuple=False) # [[batch_idx, indiv_idx], ...]
        indiv_indices = [idx.item() for _, idx in sel]
        return indiv_indices

    def store_reward(self, reward):
        """
        Store reward (float or tensor) for the most recent step.
        """
        if not torch.is_tensor(reward):
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        self.rewards.append(reward)

    def update_policy(self):
        """
        Perform REINFORCE update:
        Compute discounted returns, normalize them, and apply policy gradient.
        """
        if not self.rewards:
            return
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns).to(self.device)   # [T]
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # Compute policy loss
        policy_losses = []
        for logp, R in zip(self.log_probs, returns):
            policy_losses.append(-logp * R)
        loss = torch.stack(policy_losses).sum()
        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
