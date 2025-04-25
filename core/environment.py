# environment.py
"""
HereditaryCancerEnv with standardized tensor observations and clear separation of state.
High cohesion: handles only environment dynamics and observation encoding.
Low coupling: provides tensor obs and uses basic Python types internally.
"""
import numpy as np
import torch

class HereditaryCancerEnv:
    def __init__(
        self,
        n_individuals: int,
        M: int,
        max_time: int,
        detection_reward: float = 10.0,
        screening_cost: float = 1.0,
        false_neg_prob: dict = None,
        device: str = 'cuda'
    ):
        """
        Args:
            n_individuals: number of family members
            M: maximum tumor stage (0..M)
            max_time: episode horizon
            detection_reward: reward per true-positive detection
            screening_cost: cost per screening action
            false_neg_prob: mapping stage->false-negative probability
            device: 'cpu' or 'cuda'
        """
        self.n_individuals = n_individuals
        self.M = M
        self.max_time = max_time
        self.detection_reward = detection_reward
        self.screening_cost = screening_cost
        self.false_neg_prob = false_neg_prob or {k: 0.2 for k in range(1, M+1)}
        self.device = device

        # Observation dimension: time + (known genotype + last test) for each individual
        self.obs_dim = 1 + 2 * self.n_individuals

        # RNG for reproducibility
        self.rng = np.random.default_rng()

        # Initialize state variables
        self.reset()

    def reset(self) -> torch.Tensor:
        """
        Reset environment to initial state.
        Returns initial observation tensor of shape [1, obs_dim].
        """
        # Sample genotypes: 0 or 1 uniformly
        self.genotypes = self.rng.integers(0, 2, size=self.n_individuals).tolist()
        # Tumor stages: start healthy
        self.stages = [0] * self.n_individuals
        # Known genotype flags
        self.known_genotypes = [False] * self.n_individuals
        # Last test results (0=neg,1=pos)
        self.last_tests = [0] * self.n_individuals
        # Time step
        self.t = 0
        return self._get_obs()

    def step(self, action: list) -> tuple:
        """
        Execute screening action.
        Args:
            action: list of individual indices to screen
        Returns:
            obs: next observation tensor [1, obs_dim]
            reward: float
            done: bool
            info: dict
        """
        reward = 0.0
        # Reset test outcomes
        self.last_tests = [0] * self.n_individuals
        # Screening outcomes
        for i in action:
            stage = self.stages[i]
            # Compute positive probability
            pos_prob = 0.0 if stage == 0 else 1.0 - self.false_neg_prob.get(stage, 0.2)
            positive = self.rng.random() < pos_prob
            self.last_tests[i] = 1 if positive else 0
            if positive:
                reward += self.detection_reward
                if self.genotypes[i] == 1:
                    self.known_genotypes[i] = True
        # Subtract screening cost
        reward -= len(action) * self.screening_cost
        # Progress tumor stages
        for i in range(self.n_individuals):
            if self.stages[i] < self.M:
                # Base progression probability
                p = 0.05 * (2.0 if self.genotypes[i] == 1 else 1.0)
                if self.rng.random() < p:
                    self.stages[i] += 1
        # Advance time
        self.t += 1
        # Build next observation
        obs = self._get_obs()
        done = self.t >= self.max_time
        return obs, reward, done, {}

    def _get_obs(self) -> torch.Tensor:
        """
        Pack current state into a tensor:
        [time, gen_known_0, test_0, gen_known_1, test_1, ...]
        All floats.
        Returns tensor on specified device, shape [1, obs_dim].
        """
        arr = np.zeros(self.obs_dim, dtype=np.float32)
        arr[0] = float(self.t)
        for i in range(self.n_individuals):
            base = 1 + 2 * i
            # Known genotype: 0 if unknown, else 0/1
            arr[base] = float(self.genotypes[i]) if self.known_genotypes[i] else 0.0
            # Last test
            arr[base + 1] = float(self.last_tests[i])
        return torch.tensor(arr, device=self.device).unsqueeze(0)

    def render(self) -> None:
        """
        Debug print of hidden state.
        """
        print(f"t={self.t}, stages={self.stages}, genotypes={self.genotypes}, known={self.known_genotypes}")
