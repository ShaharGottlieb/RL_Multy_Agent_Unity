import argparse
from test import test_wrapper
from train import train_wrapper
from ddpg.ddpg_agent import Agent as DDPGAgent
from ddpg.multi_ddpg_agent import Agent as MDDPGAgent
from maddpg.maddpg_agent import Agent as MADDPGAgent
from agent import AgentABC


def select_agent(agent_type: str) -> type(AgentABC):
    """
    function for choosing agents. these are all the options. this way train and test don't care about what type of
     agent this is.
    for adding new agent - modify this function as well as --agent parser options.
    :param agent_type:
    :return: class of selected agent.
    """
    if agent_type == 'ddpg':
        agent = DDPGAgent
    elif agent_type == 'mddpg':
        agent = MDDPGAgent
    elif agent_type == 'maddpg':
        agent = MADDPGAgent
    else:
        agent = None
    return agent


def main():
    # required for test and train:
    # parsed by the main parser - this group is shared
    g_parser = argparse.ArgumentParser(add_help=False)
    g_parser.add_argument('--num-episodes', type=int,
                          help='number of running episodes (default is 1000 for train, and 5 for test')
    g_parser.add_argument('--build', default=None, type=str, required=True,
                          help='path of the unity build file, to run inside Unity - enter None')
    g_parser.add_argument('--weights-path', type=str, required=True,
                          help='path to weights dir')
    g_parser.add_argument('--agent', choices=['ddpg', 'mddpg', 'maddpg'], required=True,
                          help='type of agent')
    g_parser.add_argument('--num-agents', choices=range(1, 9), default=4, type=int, metavar='[1-8]',
                          help='number of agents (cars)')
    g_parser.add_argument('--num-obstacles', choices=range(0, 17), default=4, type=int, metavar='[0-16]',
                          help='number of random obstacles')
    # general group end
    parser = argparse.ArgumentParser(prog='RL_Multi_agent_Cars',
                                     description='please choose train or test to get specific help'
                                                 ' (e.g main.py train -h)')
    subparsers = parser.add_subparsers(help='two available running modes', dest='subparser_name')
    # define new sub-command
    subparsers.add_parser('test', help='run test mode', parents=[g_parser])
    # required for train only:
    # parse by the train command sub-parser
    train_parser = subparsers.add_parser('train', help='run train mode', parents=[g_parser])

    train_parser.add_argument('--save-mem', action='store_true',
                              help='save the replay buffer during training for later use', )
    train_parser.add_argument('--scores-avg-window', choices=range(0, 101), metavar='[0-100]', default=50, type=int,
                              help='number of last scores to average')
    train_parser.add_argument('--load-weights', action='store_true',
                              help='add this to load weights from previous runs')
    train_parser.add_argument('--load-mem', action='store_true',
                              help='add this to load replay buffer from previous run')
    train_parser.add_argument('--mem-path', type=str,
                              help='path of replay buffer file to load or store')
    train_parser.add_argument('--solved-score', default=40, type=int,
                              help='score that complete the episode')
    train_parser.add_argument('--show-graphics', action='store_true',
                              help='add this to show graphics (slows down training)')
    train_parser.add_argument('--print-agent-loss', action='store_true',
                              help='print agent\'s loss after each episode (default=False)')
    train_parser.add_argument('--save-best-weights', action='store_true',
                              help='save the best weights so far (by average score). saving directory will be the'
                                   ' same as weights-path with suffix \'best\' default=False')
    train_parser.add_argument('--save-score-log', action='store_true',
                              help='saves a csv file with the ongoing scores of each episode (default=False)')
    args = parser.parse_args()
    if args.num_episodes is None:
        args.num_episodes = 1000 if args.subparser_name == 'train' else 5

    env_config = {'num_agents': args.num_agents,
                  'num_obstacles': args.num_obstacles,
                  'setting': 0
                  }
    wrapper_config = vars(args)
    wrapper_config['agent'] = select_agent(wrapper_config['agent'])
    print('starting {} with arguments:\n{}'.format(args.subparser_name, wrapper_config))
    if args.subparser_name == 'test':
        test_wrapper(env_config, wrapper_config)
    else:
        train_wrapper(env_config, wrapper_config)


if __name__ == '__main__':
    main()
