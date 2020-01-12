from abc import ABCMeta, abstractmethod
import os


class AgentABC(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.debug_loss = 0

    @abstractmethod
    def step(self, states, actions, rewards, next_states, dones):
        """
        agent step. this is called after every step of the environment (in training).
        learning should happen here. takes as arguments the RL tuple (s,a,r,s',d)
        :param states: vector of states received from the environment (s)
        :param actions: vector of actions that were taken in current env step (a)
        :param rewards: reward vector for the current step (r)
        :param next_states: vector of next states (s')
        :param dones: vector to indicate if this was the terminal step of the episode (d)
        """
        pass

    @abstractmethod
    def act(self, state, add_noise=True):
        """
        agent act. this is called before each step of the environment (train and test)
        :param state: state to act on
        :param add_noise: whether or not to add noise to the action (train or test)
        """
        pass

    @abstractmethod
    def reset(self):
        """
        this method is called after each episode in order to reset the agent's internal state.
        """
        self.debug_loss = 0

    @abstractmethod
    def save_weights(self, directory_path):
        """
        called in order to save the current weights.
        :param directory_path: a path to the directory where to save the weights.
        """
        if not(os.path.isdir(directory_path)):
            os.mkdir(directory_path)

    @abstractmethod
    def load_weights(self, directory_path):
        """
        load existing weights (either test or continue training)
        :param directory_path: a path to the directory where to load the weights from.
        """
        if not (os.path.isdir(directory_path)):
            raise NotADirectoryError

    @abstractmethod
    def save_mem(self, directory_path):
        """
        save the training memory (replay buffer). this is useful in order to fully capture the state of the training.
        :param directory_path: a path to the directory where to save the memory.
        """
        if not (os.path.isdir(directory_path)):
            os.mkdir(directory_path)

    @abstractmethod
    def load_mem(self, directory_path):
        """
        load previous training memory (replay buffer).
        this is useful in order to fully capture the state of the training.
        :param directory_path: a path to the directory where to load the memory from.
        """
        if not (os.path.isdir(directory_path)):
            raise NotADirectoryError




