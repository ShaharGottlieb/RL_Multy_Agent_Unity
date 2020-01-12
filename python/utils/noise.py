import numpy as np
import random
import copy


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=None, theta=0.15, sigma=0.15, sigma_min=0.05, sigma_decay=.99):
        """Initialize parameters and noise process."""
        if mu is None:
            mu = [0.0, 0.3]  # add some advantage in training for hitting the gas (this makes the training faster).
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Reduce sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
