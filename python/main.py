import argparse
from test import test_wrapper
from train import train_wrapper


def main():
    # required for test and train:
    # parsed by the main parser - this group is shared
    g_parser = argparse.ArgumentParser(add_help=False)
    g_parser.add_argument('--num-episodes', type=int, help='number of running episodes')
    g_parser.add_argument('--build', default=None, type=str, help='path of the unity build file, to run inside Unity'
                                                                  ' - enter None', required=True)
    g_parser.add_argument('--weights-path', type=str, required=True,
                        help='path to weights dir')
    g_parser.add_argument('--agent', choices=['ddpg', 'mddpg', 'maddpg'], required=True,
                        help='type of agent')
    g_parser.add_argument('--num-agents', choices=range(1, 9), default=4, type=int,
                        metavar='[1-8]', help='number of agents (cars)')
    g_parser.add_argument('--num-obstacles', choices=range(0, 17), default=4, type=int,
                        metavar='[0-16]', help='number of random obstacles')
    # general group end
    parser = argparse.ArgumentParser(prog='RL_Multi_agent_Cars',
                                     description='please choose train or test to get specific help'
                                                 ' (e.g main.py train -h)')
    subparsers = parser.add_subparsers(help='two available running modes', dest='subparser_name')
    # define new subcommand
    test_parser = subparsers.add_parser('test', help='run test mode', parents=[g_parser])
    # required for train only:
    # parse by the train command subparser
    train_parser = subparsers.add_parser('train', help='run train mode', parents=[g_parser])
    train_parser.add_argument('--save-memory', default=False, help='save the replay buffer', action='store_true')
    train_parser.add_argument('--scores-avg-window', choices=range(0, 101), metavar='[0-100]', default=50,
                              type=int, help='number of last scores to average')
    train_parser.add_argument('--load-weights', default=False, action='store_true',
                              help='add this to load weights from previous runs')
    train_parser.add_argument('--load-mem', action='store_true',
                              help='add this to load replay buffer from previous run')
    train_parser.add_argument('--mem-path', type=str, required=True,
                              help='path of replay buffer file to load or store')
    train_parser.add_argument('--solved-score', default=40, type=int,
                              help='score that complete the episode')
    train_parser.add_argument('--no-graphics', type=bool, default=True,
                              help='add this to avoid graphics')
    train_parser.add_argument('--print-agent-loss', action='store_true',
                              help='print agent\'s loss after each episode (default=False)')
    train_parser.add_argument('--save-best-weights', default=True, type=bool,
                              help='save the best weights so far (by average score). saving directory will be the'
                              ' same as weights-path with suffix \'best\' default=True')
    train_parser.add_argument('--save-score-log', default=True, type=bool,
                              help='saves a csv file with the ongoing scores of each episode (default=True)')
    args = parser.parse_args()
    if args.num_episodes is None:
        args.num_episodes = 1000 if args.subparser_name == 'train' else 5
    # if args.no_graphics is None:
        # args.no_graphics = True if args.subparser_name == 'train' else False
    env_config = {'num_agents': args.num_agents,
                  'num_obstacles': args.num_obstacles,
                  'setting': 0
                  }
    wrapper_config = vars(args)
    # print(args)
    # print(env_config)
    # print(wrapper_config)
    if args.subparser_name == 'test':
        test_wrapper(env_config, wrapper_config)
    else:
        train_wrapper(env_config, wrapper_config)


if __name__ == '__main__':
    main()
