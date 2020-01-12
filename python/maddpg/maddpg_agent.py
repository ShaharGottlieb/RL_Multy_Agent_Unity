import os
import numpy as np
import random
from agent import AgentABC
from maddpg.maddpg_model import Actor, Critic
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 1        # update weights every # episodes (adds stability)
NUM_UPDATES = 1         # how many learning steps to take each learning phase (adds stability)
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1.5e-4      # learning rate of the critic
WEIGHT_DECAY = 0        # weight decay

an_filename = "maddpgActor_Model.pth"
cn_filename = "maddpgCritic_Model.pth"
memory_filename = "maddpg_memory"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AgentABC):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an MADDPG Agent object.
        Params
        ======
            :param state_size: dimension of each state
            :param action_size: dimension of each action
            :param num_agents: number of inner agents
            :param random_seed: random seed
        """
        super().__init__(state_size, action_size, num_agents, random_seed)
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        self.actors_local = []
        self.actors_target = []
        self.actor_optimizers = []
        self.critics_local = []
        self.critics_target = []
        self.critic_optimizers = []
        for i in range(num_agents):
            # Actor Network (w/ Target Network)
            self.actors_local.append(Actor(state_size, action_size, random_seed).to(device))
            self.actors_target.append(Actor(state_size, action_size, random_seed).to(device))
            self.actor_optimizers.append(optim.Adam(self.actors_local[i].parameters(), lr=LR_ACTOR))
            # Critic Network (w/ Target Network)
            self.critics_local.append(Critic(num_agents * state_size, num_agents * action_size, random_seed).to(device))
            self.critics_target.append(Critic(
                num_agents * state_size, num_agents * action_size, random_seed).to(device))
            self.critic_optimizers.append(optim.Adam(self.critics_local[i].parameters(),
                                                     lr=LR_CRITIC, weight_decay=WEIGHT_DECAY))

        # Noise process for each agent
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # debugging variables
        self.step_count = 0
        self.mse_error_list = []

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        # in order to add some stability to the learning, we don't modify weights every turn.
        self.step_count += 1
        if (self.step_count % UPDATE_EVERY) == 0:   # learn every #UPDATE_EVERY steps
            for i in range(NUM_UPDATES):            # update #NUM_UPDATES times
                if len(self.memory) > 1000:
                    experiences = self.memory.sample()
                    self.learn(experiences)
                    self.debug_loss = np.mean(self.mse_error_list)
            self.update_target_networks()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        acts = np.zeros((self.num_agents, self.action_size))
        for agent in range(self.num_agents):
            self.actors_local[agent].eval()
            with torch.no_grad():
                acts[agent, :] = self.actors_local[agent](state[agent, :]).cpu().data.numpy()
            self.actors_local[agent].train()
        if add_noise:
            acts += self.noise.sample()
        return np.clip(acts, -1, 1)

    def reset(self):
        """ see abstract class """
        super().reset()
        self.noise.reset()
        self.mse_error_list = []

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_full_state, actors_target(next_partial_state) )
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

        for agent in range(self.num_agents):
            actions_next_batched = [self.actors_target[i](next_states_batched[:, i, :]) for i in
                                    range(self.num_agents)]
            actions_next_whole = torch.cat(actions_next_batched, 1)
            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            q_targets_next = self.critics_target[agent](next_states_concated, actions_next_whole)
            # Compute Q targets for current states (y_i)
            q_targets = rewards[:, agent].view(BATCH_SIZE, -1) + (
                    GAMMA * q_targets_next * (1 - dones[:, agent].view(BATCH_SIZE, -1)))
            # Compute critic loss
            q_expected = self.critics_local[agent](states_concated, actions_concated)
            critic_loss = F.mse_loss(q_expected, q_targets)
            # Minimize the loss
            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent].step()
            # save the error for statistics
            self.mse_error_list.append(critic_loss.detach().cpu().numpy())

            # ---------------------------- update actor ---------------------------- #
            action_i = self.actors_local[agent](states_batched[:, agent, :])
            actions_pred = actions_batched.clone()
            actions_pred[:, agent, :] = action_i
            actions_pred_whole = actions_pred.view(BATCH_SIZE, -1)
            # Compute actor loss
            actor_loss = -self.critics_local[agent](states_concated, actions_pred_whole).mean()
            # Minimize the loss
            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent].step()

    def update_target_networks(self):
        # ----------------------- update target networks ----------------------- #
        for agent in range(self.num_agents):
            self.soft_update(self.critics_local[agent], self.critics_target[agent], TAU)
            self.soft_update(self.actors_local[agent], self.actors_target[agent], TAU)

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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def load_weights(self, directory_path):
        """ see abstract class """
        super().load_weights(directory_path)
        actor_weights = os.path.join(directory_path, an_filename)
        critic_weights = os.path.join(directory_path, cn_filename)
        for agent in range(self.num_agents):
            self.actors_target[agent].load_state_dict(torch.load(actor_weights + "_" + str(agent), map_location=device))
            self.critics_target[agent].load_state_dict(
                torch.load(critic_weights + "_" + str(agent), map_location=device))
            self.actors_local[agent].load_state_dict(torch.load(actor_weights + "_" + str(agent), map_location=device))
            self.critics_local[agent].load_state_dict(
                torch.load(critic_weights + "_" + str(agent), map_location=device))

    def save_weights(self, directory_path):
        """ see abstract class """
        super().save_weights(directory_path)
        actor_weights = os.path.join(directory_path, an_filename)
        critic_weights = os.path.join(directory_path, cn_filename)
        for agent in range(self.num_agents):
            torch.save(self.actors_local[agent].state_dict(), actor_weights + "_" + str(agent))
            torch.save(self.critics_local[agent].state_dict(), critic_weights + "_" + str(agent))

    def save_mem(self, directory_path):
        """ see abstract class """
        super().save_mem(directory_path)
        self.memory.save(os.path.join(directory_path, memory_filename))

    def load_mem(self, directory_path):
        """ see abstract class """
        super().load_mem(directory_path)
        self.memory.load(os.path.join(directory_path, memory_filename))
