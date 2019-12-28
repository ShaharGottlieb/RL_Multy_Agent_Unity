import numpy as np
import random
import copy
from collections import namedtuple, deque
from agent import AgentABC

from ddpg.ddpg_model import Actor, Critic
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim
import os.path

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4       # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay

an_filename = "ddpgActor_Model.pth"
cn_filename = "ddpgCritic_Model.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AgentABC):
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super().__init__(state_size, action_size, num_agents, random_seed)
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process for each agent
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for agent in range(self.num_agents):
            self.memory.add(states[agent,:], actions[agent,:], rewards[agent], next_states[agent,:], dones[agent])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        acts = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for agent in range(self.num_agents):
                acts[agent, :] = self.actor_local(state[agent, :]).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            acts += noise
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
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards.view(BATCH_SIZE, -1) + (GAMMA * Q_targets_next * (1 - dones).view(BATCH_SIZE, -1))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    @staticmethod
    def soft_update(local_model, target_model, tau):
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

    @staticmethod
    def suffix_handler(filename, suffix):    #this is so we can train single agent and the load to multi-agents
        if os.path.exists(filename+suffix):
            return filename+suffix
        elif os.path.exists(filename+str(0)):
            return filename+str(0)
        else:
            return filename

    def load_weights(self, directory_path):
        super().load_weights(directory_path)
        self.actor_target.load_state_dict(torch.load(os.path.join(directory_path, an_filename), map_location=device))
        self.critic_target.load_state_dict(torch.load(os.path.join(directory_path, cn_filename), map_location=device))
        self.actor_local.load_state_dict(torch.load(os.path.join(directory_path, an_filename), map_location=device))
        self.critic_local.load_state_dict(torch.load(os.path.join(directory_path, cn_filename), map_location=device))

    def save_weights(self, directory_path):
        super().save_weights(directory_path)
        torch.save(self.actor_local.state_dict(), os.path.join(directory_path, an_filename))
        torch.save(self.critic_local.state_dict(), os.path.join(directory_path, cn_filename))

    def save_mem(self, directory_path):
        super().save_mem(directory_path)
        self.memory.save(os.path.join(directory_path, "ddpg_memory"))

    def load_mem(self, directory_path):
        super().load_mem(directory_path)
        self.memory.load(os.path.join(directory_path, "ddpg_memory"))
