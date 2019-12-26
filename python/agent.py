from abc import ABCMeta, abstractmethod
import os


class AgentABC(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, state_size, action_size, num_agents, random_seed):
        pass

    @abstractmethod
    def step(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def act(self, state, add_noise=True):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save_weights(self, directory_path):
        if not(os.path.isdir(directory_path)):
            os.mkdir(directory_path)

    @abstractmethod
    def load_weights(self, directory_path):
        if not (os.path.isdir(directory_path)):
            raise NotADirectoryError

    @abstractmethod
    def save_mem(self, directory_path):
        if not (os.path.isdir(directory_path)):
            os.mkdir(directory_path)

    @abstractmethod
    def load_mem(self, directory_path):
        if not (os.path.isdir(directory_path)):
            raise NotADirectoryError




