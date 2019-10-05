
from ddpg.ddpg_agent import Agent as DDPGAgent

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.agents = [DDPGAgent(state_size, action_size, 1, random_seed) for i in range(num_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            states_single = states[i].reshape(1, self.state_size)
            actions_single = actions[i].reshape(1, self.action_size)
            next_states_single = next_states[i].reshape(1, self.state_size)
            dones_single = [dones[i]]
            rewards_single = [rewards[i]]
            self.agents[i].step(states_single, actions_single, rewards_single, next_states_single, dones_single)

    def act(self, state, add_noise=True):
        return [self.agents[i].act(state[i].reshape(1,self.state_size), add_noise) for i in range(self.num_agents)]

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def LoadWeights(self):
        for agent in range(self.num_agents):
            self.agents[agent].LoadWeights(str(agent))

    def SaveWeights(self):
        for agent in range(self.num_agents):
            self.agents[agent].SaveWeights(str(agent))

    def SaveMem(self):
        for agent in range(self.num_agents):
            self.agents[agent].SaveMem(str(agent))

    def LoadMem(self):
        for agent in range(self.num_agents):
            self.agents[agent].LoadMem(str(agent))

