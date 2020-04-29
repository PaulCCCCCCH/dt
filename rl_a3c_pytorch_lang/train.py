from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads, find_closest
import numpy as np
import random
from scipy import spatial

from model_lang import A3Clstm
# from model import A3Clstm

from player_util_lang import Agent
from torch.autograd import Variable


def train(rank, args, shared_model, optimizer, env_conf, emb, bi_grams, instructions):
    # Changes the process name
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)

    # Define special vectors
    eos_vector = emb.get_vector("<eos>")
    oov_vector = emb.get_vector("<oov>")

    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)

    # Create agent
    player = Agent(None, env, args, None, emb)
    player.gpu_id = gpu_id

    # Create DNN model for the agent
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space, emb)

    # Set env and move to gpu
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()

    # Set model to "training" mode. Not doing anything but is a good practice to add
    player.model.train()

    # Start iteration
    player.eps_len += 2

    _counter = 0
    while True:

        # Loading param values from shared model
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())

        # Reset LSTM state when episode ends
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, args.lstm_size).cuda())
                    player.hx = Variable(torch.zeros(1, args.lstm_size).cuda())
            else:
                player.cx = Variable(torch.zeros(1, args.lstm_size))
                player.hx = Variable(torch.zeros(1, args.lstm_size))

        # If not ended, save current state value
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        # Make a step and record observations. Repeat until num_steps reached or game is over.
        for step in range(args.num_steps):
            # Random period alpha (Changing alpha only after running for a period)
            if step % 100 == 0:
                r = random.random()
                if r < 0.33:
                    player.model.alpha = 1
                elif r < 0.67:
                    player.model.alpha = 0.5
                else:
                    player.model.alpha = 0
            player.action_train()
            if player.done:
                break

        # If episode finished before args.num_steps is reached, reset environment
        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        # If episode not finished after args.num_steps:
        # Estimates value function of current state
        R = torch.zeros(1, 1)
        if not player.done:
            _, value, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        # Append reward for the final time step
        player.values.append(Variable(R))

        # Initialise loss accumulator
        policy_loss = 0
        value_loss = 0
        language_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)

        # Accumulate the losses
        for i in reversed(range(len(player.rewards))):

            # Calculating language loss
            if args.use_language:

                # Calculating language loss
                # Get action of a time step
                a = np.argmax(player.action_logits[i].detach().cpu().numpy())

                # Get produced vectors of the time step
                produced_logits = player.produced_logits[i]
                # print(produced_vectors)
                # Get target vectors of the time step (an instruction corresponding to the least cost)
                action_instructions = instructions[a]

                # Sample a few from the set
                for _ in range(10):
                    idx = random.randrange(0, len(action_instructions))
                    instruction = action_instructions[idx]


                    target_words = instruction.split()

                    for pos, target_word in enumerate(target_words):
                        target_class = torch.tensor(emb.get_index(target_word)).cuda()
                        produced_logit = produced_logits[pos]

                        # Cross_entropy combines log-softmax and nll
                        # Here procuded_vec is one-hot while target is an integer
                        language_loss += torch.nn.functional.cross_entropy(produced_logit, target_class.unsqueeze(0))
                        if target_word == '<eos>':
                            break


            # Calculate other losses
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]

            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]


        # Initialise grad accumulator
        player.model.zero_grad()

        # Calculate grad and update
        if args.use_language:
            (policy_loss + 0.5 * value_loss + 0.1 * 0.01* language_loss).backward()
        else:
            (policy_loss + 0.5 * value_loss).backward()

        """
        # (policy_loss + 0.5 * value_loss).backward()
        print("****************")
        print(policy_loss)
        print(value_loss)
        # """
        if args.use_language and _counter % 10 == 0:
            print("****************")
            #print(policy_loss)
            #print(value_loss)
            print("language loss", language_loss)
        _counter += 1

        # Copying over the parameters to shared model
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()

        # Clean agent observations
        player.clear_actions()
