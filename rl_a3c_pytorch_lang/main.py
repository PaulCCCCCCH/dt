from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import atari_env
from utils import read_config, read_pong_instructions
# from model import A3Clstm
from model_lang import A3Clstm
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
#from gym.configuration import undo_logger_setup
import time
from embedding import Embedding
import numpy as np
from params import args

# Define arguments
#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
        if type(args.gpu_ids) == int:
            args.gpu_ids = [args.gpu_ids]

    # Defines crop range for the picked environment
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)

    # Reading instruction file
    instructions, specific_vocab, bi_grams = read_pong_instructions("./data/pong.txt")

    # Read embedding model
    if args.use_full_emb:
        import gensim
        emb = gensim.models.KeyedVectors.load_word2vec_format(args.emb_path, binary=True, limit=10000)


    else:
        emb = Embedding(args.emb_path, specific_vocab)

    # Append special words to the embedding model
    direction = np.zeros(args.emb_dim)
    direction[0] = 1
    emb.add("<eos>", direction) # ignore the warning here
    direction[0] = 0
    direction[1] = 1
    emb.add("<pad>", direction)
    direction[1] = 0
    direction[2] = 1
    emb.add("<oov>", direction)
    direction[2] = 0
    direction[3] = 1
    emb.add("<sos>", direction)


    # Creates a shared model and load from checkpoint
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space, emb)
    if args.load:
        # Get the (var, val) dict
        saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.env),
            map_location=lambda storage, loc: storage)
        # Restore the variable dictionary
        model_state_dict = shared_model.state_dict()
        for k, v in saved_state.items():
            model_state_dict.update({k: v})
        shared_model.load_state_dict(model_state_dict)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    # Start tester
    p = mp.Process(target=test, args=(args, shared_model, env_conf, emb))
    p.start()
    processes.append(p)
    time.sleep(0.1)

    # Start seperate workers
    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf, emb, bi_grams))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
