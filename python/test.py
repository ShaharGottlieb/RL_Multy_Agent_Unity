###################################
# Import Required Packages
import torch
import time
import random
import os
import numpy as np
from ddpg.ddpg_agent import Agent as DDPGAgent
from ddpg.multi_ddpg_agent import Agent as MDDPGAgent
from maddpg.maddpg_agent import Agent as MADDPGAgent
from mlagents.envs import UnityEnvironment

"""
###################################
STEP 1: Set the Test Parameters
======
        num_episodes (int): number of test episodes
"""
num_episodes=5
env_config = {"num_agents": 5, "setting": 0, "num_obstacles": 6}

"""
###################################
STEP 2: Start the Unity Environment
# Use the corresponding call depending on your operating system 
"""
#env = UnityEnvironment(file_name=None)
env = UnityEnvironment(file_name=os.path.join("build_race","OurProject.exe"))
# - **Mac**: "Banana_Mac/Reacher.app"
# - **Windows** (x86): "Reacher_Windows_x86/Reacher.exe"
# - **Windows** (x86_64): "Reacher_Windows_x86_64/Reacher.exe"
# - **Linux** (x86): "Reacher_Linux/Reacher.x86"
# - **Linux** (x86_64): "Reacher_Linux/Reacher.x86_64"
# - **Linux** (x86, headless): "Reacher_Linux_NoVis/Reacher.x86"
# - **Linux** (x86_64, headless): "Reacher_Linux_NoVis/Reacher.x86_64"

"""
#######################################
STEP 3: Get The Unity Environment Brian
Unity ML-Agent applications or Environments contain "BRAINS" which are responsible for deciding 
the actions an agent or set of agents should take given a current set of environment (state) 
observations. The Reacher environment has a single Brian, thus, we just need to access the first brain 
available (i.e., the default brain). We then set the default brain as the brain that will be controlled.
"""
# Get the default brain 
brain_name = env.brain_names[0]

# Assign the default brain as the brain to be controlled
brain = env.brains[brain_name]


"""
#############################################
STEP 4: Determine the size of the Action and State Spaces and the Number of Agents

The observation space consists of 33 variables corresponding to
position, rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque 
applicable to two joints. Every entry in the action vector should 
be a number between -1 and 1.

The reacher environment can contain multiple agents in the environment to increase training time. 
To use multiple (active) training agents we need to know how many there are.
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
###################################
STEP 5: Initialize a DDPG Agent from the Agent Class in dqn_agent.py
A DDPG agent initialized with the following parameters.
    ======
    state_size (int): dimension of each state (required)
    action_size (int): dimension of each action (required)
    num_agents (int): number of agents in the unity environment
    seed (int): random seed for initializing training point (default = 0)

Here we initialize an agent using the Unity environments state and action size and number of Agents
determined above.
"""
#Initialize Agent
#agent = DDPGAgent(state_size=state_size, action_size=action_size[0], num_agents=num_agents, random_seed=0)
#agent = MADDPGAgent(state_size=state_size, action_size=action_size[0], num_agents=num_agents, random_seed=0)
agent = MDDPGAgent(state_size=state_size, action_size=action_size[0], num_agents=num_agents, random_seed=0)

# Load trained model weights
agent.LoadWeights()
"""
###################################
STEP 6: Play Banana for specified number of Episodes
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
        dones = env_info.local_done                  # see if episode has finished for each unity agent in the environment

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
###################################
STEP 7: Everything is Finished -> Close the Environment.
"""
env.close()

# END :) #############

