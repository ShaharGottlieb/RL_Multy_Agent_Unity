###################################
# Import Required Packages
import numpy as np
import os
from mlagents.envs import UnityEnvironment
from agent import AgentABC


def test_wrapper(env_config, wrapper_config):
    """
    Set the Test Parameters
    """
    # num_episodes (int): number of test episodes
    num_episodes = wrapper_config['num_episodes']

    # build_path: path to the build of the unity environment.
    build = None if wrapper_config['build'] == 'None' else wrapper_config['build']
    if (build is not None) and (not os.path.isfile(build)):
        print('--build is not a valid path')
        raise FileNotFoundError

    # weights_path: path to the directory containing the weights (same directory to save them)
    weights_path = wrapper_config['weights_path']
    if not os.path.isdir(weights_path):
        print('--weights-path is not a valid directory')
        raise NotADirectoryError

    # agent_type (DDPG | MDDPG | MADDPG)
    agent_type = wrapper_config['agent']
    if not issubclass(agent_type, AgentABC):
        print('invalid agent type')
        raise TypeError

    """
    Start the Unity Environment
    """
    env = UnityEnvironment(file_name=build)

    """
    Get The Unity Environment Brain
    Unity ML-Agent applications or Environments contain "BRAINS" which are responsible for deciding 
    the actions an agent or set of agents should take given a current set of environment (state) 
    observations. The Race environment has a single Brain, thus, we just need to access the first brain 
    available (i.e., the default brain). We then set the default brain as the brain that will be controlled.
    """
    # Get the default brain
    brain_name = env.brain_names[0]

    # Assign the default brain as the brain to be controlled
    brain = env.brains[brain_name]

    """
    Determine the size of the Action and State Spaces and the Number of Agents.
    The observation space consists of variables corresponding to Ray Cast in different direction, 
    velocity and direction.  
    Each action is a vector with 2 numbers, corresponding to steer left/right and brake/drive (in this order).
    each action is a number between -1 and 1.
    num_agents will correspond to the number of agent using the same brain -
    (since all cars use the same action / observation space they all use the same brain)
    if in the future one should have different cars use different observation space, 
    one will need to split them into different brains..
    """
    # Set the number of actions or action size
    action_size = brain.vector_action_space_size

    # Set the size of state observations or state size
    state_size = brain.vector_observation_space_size

    # Get number of agents in Environment
    env_info = env.reset(train_mode=True, config=env_config)[brain_name]
    num_agents = len(env_info.agents)
    print('\nNumber of Agents: ', num_agents)

    """
    Initialize an Agent from the Agent Class in Agent.py
    Any agent initialized with the following parameters.
        ======
        state_size (int): dimension of each state (required)
        action_size (int): dimension of each action (required)
        num_agents (int): number of agents in the unity environment
        seed (int): random seed for initializing training point (default = 0)

    Here we initialize an agent using the Unity environments state and action size and number of Agents
    determined above.
    TODO - agent type check, add type hints to abstract class.
    """
    agent: AgentABC = agent_type(state_size=state_size,
                                 action_size=action_size[0], num_agents=num_agents, random_seed=0)

    # Load trained model weights
    agent.load_weights(weights_path)
    """
    Run test for number of episodes
    """
    # loop from num_episodes
    for i_episode in range(1, num_episodes+1):

        # reset the unity environment at the beginning of each episode
        # set train mode to false
        env_info = env.reset(train_mode=False, config=env_config)[brain_name]

        # get initial state of the unity environment
        states = env_info.vector_observations

        # reset the training agent for new episode
        agent.reset()

        # set the initial episode scores to zero for each unity agent.
        scores = np.zeros(num_agents)

        # Run the episode loop;
        # At each loop step take an action as a function of the current state observations
        # If environment episode is done, exit loop...
        # Otherwise repeat until done == true
        while True:
            # determine actions for the unity agents from current sate
            actions = agent.act(states, add_noise=False)

            # send the actions to the unity agents in the environment and receive resultant environment information
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations   # get the next states for each unity agent in the environment
            rewards = env_info.rewards                   # get the rewards for each unity agent in the environment
            dones = env_info.local_done           # see if episode has finished for each unity agent in the environment

            # set new states to current states for determining next actions
            states = next_states

            # Update episode score for each unity agent
            scores += rewards

            # If any unity agent indicates that the episode is done,
            # then exit episode loop, to begin new episode
            if np.any(dones):
                break

        # Print current average score
        print('\nEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores), end=""))

    """
    Everything is Finished -> Close the Environment.
    """
    env.close()

    # END :) #############

