import numpy as np
import random
import copy
from collections import namedtuple, deque

from maddpg.maddpg_model import Actor, Critic
from utils.replay_buffer import ReplayBuffer
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1.5e-4       # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay

an_filename = "maddpgActor_Model.pth"
cn_filename = "maddpgCritic_Model.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actors_local = [Actor(state_size, action_size, random_seed).to(device) for i in range(num_agents)]
        self.actors_target = [Actor(state_size, action_size, random_seed).to(device) for i in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=LR_ACTOR) for actor in self.actors_local]

        # Critic Network (w/ Target Network)
        self.critics_local = [Critic(num_agents*state_size, num_agents*action_size, random_seed).to(device)
                              for i in range(num_agents)]
        self.critics_target = [Critic(num_agents*state_size, num_agents*action_size, random_seed).to(device)
                              for i in range(num_agents)]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
                                  for critic in self.critics_local]

        # Noise process for each agent
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        acts = np.zeros((self.num_agents, self.action_size))
        for agent in range(self.num_agents):
            self.actors_local[agent].eval()
            with torch.no_grad():
                acts[agent,:] = self.actors_local[agent](state[agent,:]).cpu().data.numpy()
            self.actors_local[agent].train()
        if add_noise:
            acts += self.noise.sample()
        return np.clip(acts, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states_batched, actions_batched, rewards, next_states_batched, dones = experiences
        states_concated = states_batched.view([BATCH_SIZE, self.num_agents * self.state_size])
        next_states_concated = next_states_batched.view([BATCH_SIZE, self.num_agents * self.state_size])
        actions_concated = actions_batched.view([BATCH_SIZE, self.num_agents * self.action_size])

        # states, actions, rewards, next_states, dones = experiences
        # reshape to select partial obeservation
        # next_states_batched = next_states.reshape(BATCH_SIZE, self.num_agents, self.state_size)
        # states_batched = states.reshape(BATCH_SIZE, self.num_agents, self.state_size)
        # actions_batched = actions.reshape(BATCH_SIZE, self.num_agents, self.action_size)
        for agent in range(self.num_agents):
            actions_next_batched = [self.actors_target[agent](next_states_batched[:, agent, :]) for agent in
                                        range(self.num_agents)]
            actions_next_whole = torch.cat(actions_next_batched, 1)
            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            Q_targets_next = self.critics_target[agent](next_states_concated, actions_next_whole)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards[:,agent].view(BATCH_SIZE,-1) + (GAMMA * Q_targets_next * (1 - dones[:,agent].view(BATCH_SIZE,-1)))
            # Compute critic loss
            Q_expected = self.critics_local[agent](states_concated, actions_concated)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent].step()

            # ---------------------------- update actor ---------------------------- #
            action_i = self.actors_local[agent](states_batched[:, agent, :])
            actions_pred = actions_batched.clone()
            actions_pred[:,agent,:] = action_i
            actions_pred_whole = actions_pred.view(BATCH_SIZE,-1)
            # Compute actor loss
            actor_loss = -self.critics_local[agent](states_concated, actions_pred_whole).mean()
            # Minimize the loss
            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent].step()

        # ----------------------- update target networks ----------------------- #
        for agent in range(self.num_agents):
            self.soft_update(self.critics_local[agent], self.critics_target[agent], TAU)
            self.soft_update(self.actors_local[agent], self.actors_target[agent], TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def LoadWeights(self):
        for agent in range(self.num_agents):
            self.actors_target[agent].load_state_dict(torch.load(an_filename+"_"+str(agent), map_location=device))
            self.critics_target[agent].load_state_dict(torch.load(cn_filename+"_"+str(agent), map_location=device))
            self.actors_local[agent].load_state_dict(torch.load(an_filename+"_"+str(agent), map_location=device))
            self.critics_local[agent].load_state_dict(torch.load(cn_filename+"_"+str(agent), map_location=device))

    def SaveWeights(self):
        for agent in range(self.num_agents):
            torch.save(self.actors_local[agent].state_dict(), an_filename+"_"+str(agent))
            torch.save(self.critics_local[agent].state_dict(), cn_filename+"_"+str(agent))

    def SaveMem(self):
        self.memory.save("maddpg_memory")

    def LoadMem(self):
        self.memory.load("maddpg_memory")


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=[0.0,0.3], theta=0.15, sigma=0.15, sigma_min = 0.05, sigma_decay=.99):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
