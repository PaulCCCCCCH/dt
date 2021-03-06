
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='A3C_EVAL')
    parser.add_argument(
        '--env',
        default='Pong-v0',
        metavar='ENV',
        help='environment to train on (default: Pong-v0)')
    parser.add_argument(
        '--env-config',
        default='config.json',
        metavar='EC',
        help='environment to crop and resize info (default: config.json)')
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        metavar='NE',
        help='how many episodes in evaluation (default: 100)')
    parser.add_argument(
        '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
    parser.add_argument(
        '--render-freq',
        type=int,
        default=1,
        metavar='RF',
        help='Frequency to watch rendered game play')
    parser.add_argument(
        '--rand-gen',
        action='store_true',
        help='Randomise language generation')
    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=10000,
        metavar='M',
        help='maximum length of an episode (default: 100000)')
    parser.add_argument(
        '--new-gym-eval',
        default=False,
        metavar='NGE',
        help='Create a gym evaluation for upload')
    parser.add_argument(
        '--manual-control',
        action='store_true',
        help='Enabling manual control. Only works during evaluation')
    parser.add_argument(
        '--render',
        action='store_true',
        help='Watch game as it being played')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        metavar='LR',
        help='learning rate (default: 0.0001)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        metavar='G',
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--tau',
        type=float,
        default=1.00,
        metavar='T',
        help='parameter for GAE (default: 1.00)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--workers',
        type=int,
        default=32,
        metavar='W',
        help='how many training processes to use (default: 32)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=20,
        metavar='NS',
        help='number of forward steps in A3C (default: 20)')
    parser.add_argument(
        '--shared-optimizer',
        default=True,
        metavar='SO',
        help='use an optimizer without shared statistics.')
    parser.add_argument(
        '--load', action='store_true', help='load a trained model')
    parser.add_argument(
        '--save-max',
        default=True,
        metavar='SM',
        help='Save model on every test run high score matched or bested')
    parser.add_argument(
        '--optimizer',
        default='Adam',
        metavar='OPT',
        help='shares optimizer choice of Adam or RMSprop')
    parser.add_argument(
        '--load-model-dir',
        default='trained_models/',
        metavar='LMD',
        help='folder to load trained models from')
    parser.add_argument(
        '--save-model-dir',
        default='trained_models/',
        metavar='SMD',
        help='folder to save trained models')
    parser.add_argument(
        '--emb-path', default='', metavar='EP', help='embedding file to load')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        default=0,
        nargs='+',
        help='GPUs to use [-1 CPU only] (default: -1)')

    parser.add_argument(
        '--amsgrad',
        default=True,
        metavar='AM',
        help='Adam optimizer amsgrad parameter')
    parser.add_argument(
        '--skip-rate',
        type=int,
        default=4,
        metavar='SR',
        help='frame skip rate (default: 4)')
    parser.add_argument(
        '--use-full-emb',
        action='store_true'
    )
    parser.add_argument(
        '--use-language',
        action='store_true'
    )
    parser.add_argument(
        '--emb-dim',
        type=int,
        default=25
    )

    parser.add_argument(
        '--emb-to-load',
        type=int,
        default=300,
    )
    parser.add_argument(
        '--lstm-size',
        type=int,
        default=100
    )

    parser.add_argument(
        '--lm-dir',
        default='',
        help="Directory of the pretrained language model"
    )
    parser.add_argument(
        '--alpha-mode',
        default='none',
        choices=["none", "step", "period", "episode"],
        help='Mode of randomising alpha'
    )

    #########################################################
    # The following ones are for pre-training language model#
    #########################################################
    parser.add_argument(
        '--use-ckpt',
        action='store_true'
    )
    parser.add_argument(
        '--eval',
        action='store_true'
    )

    args = parser.parse_args()
    return args

args = get_args()