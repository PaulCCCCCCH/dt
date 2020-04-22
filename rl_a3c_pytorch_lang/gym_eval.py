from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from environment import atari_env
from utils import read_config, setup_logger
from model_lang import A3Clstm
from player_util_lang import Agent
import gym
import logging
import time
#from gym.configuration import undo_logger_setup
import numpy as np
from embedding import Embedding
from params import args

#undo_logger_setup()


setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]

gpu_id = args.gpu_id

torch.manual_seed(args.seed)
if gpu_id >= 0:
    torch.cuda.manual_seed(args.seed)

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.env),
    map_location=lambda storage, loc: storage)

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger('{}_mon_log'.format(
    args.env))

d_args = vars(args)
for k in d_args.keys():
    log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

# Read embedding model
if args.use_full_emb:
    import gensim

    emb = gensim.models.KeyedVectors.load_word2vec_format(args.emb_path, binary=True, limit=10000)

    # Append special words to the embedding model
    direction = np.zeros(100)

    direction[0] = 1
    emb.add("<eos>", direction)  # ignore the warning here
    direction[1] = 1
    emb.add("<pad>", direction)
    direction[2] = 1
    emb.add("<oov>", direction)

else:
    emb = Embedding(args.emb_path)
    emb.add_word("<eos>")
    emb.add_word("<pad>")
    emb.add_word("<oov>", np.ones(emb.emb_dim))

env = atari_env("{}".format(args.env), env_conf, args)
num_tests = 0
start_time = time.time()
reward_total_sum = 0
player = Agent(None, env, args, None, emb)
player.model = A3Clstm(player.env.observation_space.shape[0],
                       player.env.action_space, emb)
player.gpu_id = gpu_id
if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()
if args.new_gym_eval:
    player.env = gym.wrappers.Monitor(
        player.env, "{}_monitor".format(args.env), force=True)

if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model.load_state_dict(saved_state)
else:
    player.model.load_state_dict(saved_state)

player.model.eval()
for i_episode in range(args.num_episodes):
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
    player.eps_len += 2
    reward_sum = 0
    while True:
        if args.render:
            if i_episode % args.render_freq == 0:
                player.env.render()

        player.action_test()
        reward_sum += player.reward

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_mon_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))
            player.eps_len = 0
            break
