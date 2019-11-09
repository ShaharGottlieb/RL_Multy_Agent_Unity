import numpy as np

angle_90 = 3
angle_85 = 1
angle_95 = 5
angle_75 = 9
angle_105 = 11
angle_60 = 7
angle_120 = 13
angle_45 = 17
angle_135 = 19

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


    def step(self, states, actions, rewards, next_states, dones):
        return

    def act(self, state, add_noise=True):
        actions = []
        for i in range(self.num_agents):
            action = np.zeros([2], dtype=float)
            action[1] = 1
            if state[i, angle_85] > state[i, angle_95]:
                action[0] += -state[i, angle_85] * 0.3
            else:
                action[0] += state[i, angle_95] * 0.3

            if state[i, angle_75] > state[i, angle_105]:
                action[0] += -state[i, angle_75] * 0.5
            else:
                action[0] += state[i, angle_105] * 0.5

            if state[i, angle_60] > state[i, angle_120]:
                action[0] += -state[i, angle_60] * 0.7
            else:
                action[0] += state[i, angle_120] * 0.7

            if (state[i, angle_90] > 0.6) and (state[i, angle_90] > state[i, angle_85]) and (state[i, angle_90] > state[i, angle_95]):
                action[0] *= 10

            #if state[i, angle_45] > state[i, angle_135]:
            #    action[0] += -state[i, angle_45] * 0.9
            #else:
            #    action[0] += state[i, angle_135] * 0.9




            actions.append(action)
        actions = np.array(actions)
        return np.clip(actions, -1, 1)

    def reset(self):
        return

    def LoadWeights(self, suffix=""):
        return

    def SaveWeights(self, suffix=""):
        return

    def SaveMem(self, suffix=""):
        return

    def LoadMem(self, suffix=""):
        return

