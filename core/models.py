# models.py
"""
Defines abstract policy and risk model interfaces plus concrete implementations.
High cohesion: models.py only contains network definitions and factories.
Low coupling: networks are interchangeable via factory functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Policy Networks
# -----------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    Abstract base class for screening policy networks.
    Input: feature tensor of shape [batch_size, feat_dim]
    Output: logits tensor of shape [batch_size, n_individuals]
    """
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardPolicy(PolicyNetwork):
    """
    Simple MLP: feat_dim -> hidden -> n_individuals logits
    """
    def __init__(self, input_dim: int, n_individuals: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_individuals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h)

class RecurrentPolicy(PolicyNetwork):
    """
    RNN-based policy: embeds features temporally.
    """
    def __init__(
        self,
        input_dim: int,
        n_individuals: int,
        hidden_size: int = 64,
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, n_individuals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: either [batch, feat_dim] or [batch, seq_len, feat_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # treat as seq_len=1
        rnn_out, _ = self.rnn(x)
        h_last = rnn_out[:, -1, :]
        return self.fc_out(h_last)

class TransformerPolicy(PolicyNetwork):
    """
    Mini-transformer policy: self-attended tokens pooled to logits.
    """
    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        n_individuals: int,
        num_heads: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(n_tokens, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, n_individuals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_tokens] or [batch, n_tokens, d_model]
        if x.dim() == 2:
            x = self.token_embedding(x.long())
        x = x + self.pos_embedding[:, : x.size(1), :]
        encoded = self.transformer(x)
        pooled = encoded.mean(dim=1)
        return self.fc_out(pooled)

# -----------------------------------------------------------------------------
# Risk Models
# -----------------------------------------------------------------------------
class RiskModel(nn.Module):
    """
    Abstract base class for risk assessment models.
    Input: raw observation tensor [batch_size, obs_dim]
    Output: feature tensor [batch_size, feat_dim]
    """
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardRisk(RiskModel):
    """
    MLP-based risk model: obs_dim -> hidden -> feat_dim
    """
    def __init__(self, obs_dim: int, feat_dim: int, hidden_size: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, feat_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(obs))
        return F.relu(self.fc2(h))

class RecurrentRisk(RiskModel):
    """
    RNN-based risk model: processes obs as time sequence.
    """
    def __init__(
        self,
        obs_dim: int,
        feat_dim: int,
        hidden_size: int = 64,
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(obs_dim, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(obs_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, feat_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [batch, obs_dim] or [batch, seq_len, obs_dim]
        if obs.dim() == 2:
            x = obs.unsqueeze(1)
        else:
            x = obs
        out, _ = self.rnn(x)
        h_last = out[:, -1, :]
        return F.relu(self.fc(h_last))

# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------
def make_policy_network(policy_type: str, **kwargs) -> PolicyNetwork:
    """
    Instantiate a policy network by name.
    """
    if policy_type == 'mlp':
        return FeedForwardPolicy(
            input_dim=kwargs['input_dim'],
            n_individuals=kwargs['n_individuals'],
            hidden_size=kwargs.get('hidden_size', 64)
        )
    elif policy_type == 'rnn':
        return RecurrentPolicy(
            input_dim=kwargs['input_dim'],
            n_individuals=kwargs['n_individuals'],
            hidden_size=kwargs.get('hidden_size', 64),
            rnn_type=kwargs.get('rnn_type', 'LSTM')
        )
    elif policy_type == 'transformer':
        return TransformerPolicy(
            n_tokens=kwargs.get('n_tokens', 10),
            d_model=kwargs.get('d_model', 64),
            n_individuals=kwargs['n_individuals'],
            num_heads=kwargs.get('num_heads', 2),
            num_layers=kwargs.get('num_layers', 2),
            dim_feedforward=kwargs.get('dim_feedforward', 128),
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown policy_type {policy_type}")


def make_risk_model(risk_type: str, **kwargs) -> RiskModel:
    """
    Instantiate a risk model by name.
    """
    if risk_type == 'feed':
        return FeedForwardRisk(
            obs_dim=kwargs['obs_dim'],
            feat_dim=kwargs['feat_dim'],
            hidden_size=kwargs.get('hidden_size', 32)
        )
    elif risk_type == 'rnn':
        return RecurrentRisk(
            obs_dim=kwargs['obs_dim'],
            feat_dim=kwargs['feat_dim'],
            hidden_size=kwargs.get('hidden_size', 64),
            rnn_type=kwargs.get('rnn_type', 'LSTM')
        )
    else:
        raise ValueError(f"Unknown risk_type {risk_type}")
