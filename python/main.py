import argparse
import sys
from test import test_wrapper
from train import train_wrapper


def main():
    parser = argparse.ArgumentParser(prog='RL_Multi_agent_Cars',
                                     description='you can config this environment using this options')
    parser.add_argument('-m', '--mode', choices=['train', 'test'], required=True,
                        metavar='<train|test>', help='train or test')
    parser.add_argument('--num-agents', choices=range(1, 9), default=4, type=int,
                        metavar='[1-8]', help='number of agents (cars)')
    parser.add_argument('--num-obstacles', choices=range(0, 17), default=4, type=int,
                        metavar='[0-16]', help='number of random obstacles')
    parser.add_argument('--name-weights', default='file_name', type=str, help='name of the weights file')
    parser.add_argument('--save-memory', default=False, help='save the replay buffer', action='store_true')
    parser.add_argument('--num-episodes', help='number of running episodes')
    parser.add_argument('--env-setting', default=0, choices=range(0, 0), help='TODO')
    parser.add_argument('--build-path', required=True, type=str, help='path of the unity build file')
    parser.add_argument('--scores-avg-window', choices=range(0, 101), metavar='[0-100]', default=50,
                        type=int, help='number of last scores to average')
    parser.add_argument('--load-weights', default=False, action='store_true',
                        help='add this to load weights from previous runs')
    parser.add_argument('--load-weights-file', type=str, required='--load-weights' in sys.argv,
                        help='path of the weights file (if needed)')
    parser.add_argument('--load-mem', default=False, action='store_true',
                        help='add this to load replay buffer from previous run')
    parser.add_argument('--load-mem-file', type=str, required='--load-mem' in sys.argv,
                        help='path of replay buffer file to load (if needed)')
    parser.add_argument('--solved-score', default=40, type=int,
                        help='score that complete the episode')
    parser.add_argument('--no-graphics', default=False, action='store_true',
                        help='add this to avoid graphics while training')
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
    test_config = {'num_episodes': args.num_episodes,
                   'build_path': args.build_path
                   }
    train_config = {'num_episodes': args.num_episodes,
                    'build_path': args.build_path,
                    'scores_avarage_window': args.scores_avg_window,
                    'solved_score': args.solved_score,
                    'load_weights': args.load_weights,
                    'load_mem': args.load_mem,
                    'load_weights_path': args.load_weights_file,
                    'load_mem_path': args.load_mem_file,
                    'no_graphics': args.no_graphics
                    }
    print(env_config)
    print(test_config)
    print(train_config)
    if args.mode == 'test':
        test_wrapper(env_config, test_config)
    else:
        train_wrapper(env_config, train_config)


if __name__ == '__main__':
    main()

