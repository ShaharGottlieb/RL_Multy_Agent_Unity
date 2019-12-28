import argparse
import sys
from test import test_wrapper
from train import train_wrapper

# /Users/rotemlevinson/RL_project_unity/build1.app

def main():
    parser = argparse.ArgumentParser(prog='RL_Multi_agent_Cars',
                                     description='you can config this environment using this options')
    parser.add_argument('-m', '--mode', choices=['train', 'test'], required=True,
                        metavar='<train|test>', help='train or test')
    parser.add_argument('--num-agents', choices=range(1, 9), default=4, type=int,
                        metavar='[1-8]', help='number of agents (cars)')
    parser.add_argument('--num-obstacles', choices=range(0, 17), default=4, type=int,
                        metavar='[0-16]', help='number of random obstacles')
    parser.add_argument('--save-memory', default=False, help='save the replay buffer', action='store_true')
    parser.add_argument('--num-episodes', help='number of running episodes')
    parser.add_argument('--env-setting', default=0, choices=range(0, 0), help='TODO')
    parser.add_argument('--build', default=None, type=str, help='path of the unity build file')
    parser.add_argument('--scores-avg-window', choices=range(0, 101), metavar='[0-100]', default=50,
                        type=int, help='number of last scores to average')
    parser.add_argument('--load-weights', default=False, action='store_true',
                        help='add this to load weights from previous runs')
    parser.add_argument('--weights-path', type=str, required=True,
                        help='path to weights dir')
    parser.add_argument('--load-mem', default=False, action='store_true',
                        help='add this to load replay buffer from previous run')
    parser.add_argument('--mem-path', type=str, required=True,
                        help='path of replay buffer file to load or store')
    parser.add_argument('--solved-score', default=40, type=int,
                        help='score that complete the episode')
    parser.add_argument('--no-graphics', default=False, action='store_true',
                        help='add this to avoid graphics while training')
    parser.add_argument('--agent', choices=['ddpg', 'mddpg', 'maddpg'], required=True,
                        help='type of agent')
    args = parser.parse_args()
    if args.num_episodes is None:
        if args.mode == 'train':
            args.num_episodes = 1000
        else:
            args.num_episodes = 5
    env_config = {'num_agents': args.num_agents,
                  'num_obstacles': args.num_obstacles,
                  'setting': args.env_setting
                  }
    wrapper_config = {'num_episodes': args.num_episodes,
                      'build': args.build,
                      'scores_avg_window': args.scores_avg_window,
                      'solved_score': args.solved_score,
                      'load_weights': args.load_weights,
                      'load_mem': args.load_mem,
                      'weights_path': args.weights_path,
                      'mem_path': args.mem_path,
                      'no_graphics': args.no_graphics,
                      'agent_type': args.agent
                    }
    print(env_config)
    print(wrapper_config)
    if args.mode == 'test':
        test_wrapper(env_config, wrapper_config)
    else:
        train_wrapper(env_config, wrapper_config)


if __name__ == '__main__':
    main()

