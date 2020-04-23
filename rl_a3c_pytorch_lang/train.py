from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
import numpy as np
from scipy import spatial

from model_lang import A3Clstm
# from model import A3Clstm

from player_util_lang import Agent
from torch.autograd import Variable


def train(rank, args, shared_model, optimizer, env_conf, emb, instructions):
    # Changes the process name
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)

    # Define <eos> (end-of-sentence) vector
    eos_vector = emb.get_vector("<eos>")

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

            if args.use_language:
                # Calculating language loss
                # Get action of a time step
                a = np.argmax(player.log_probs[i].detach().cpu().numpy())

                # Get produced vectors of the time step
                produced_vectors = player.produced_vectors[i]
                # print(produced_vectors)
                # Get target vectors of the time step (an instruction corresponding to the least cost)
                action_instructions = instructions[a]
                losses = torch.empty(len(action_instructions)).cuda()
                for idx, instruction in enumerate(action_instructions):

                    # print(a, instruction)
                    ins_words = instruction.split()
                    # Building target vector
                    target_vectors = []
                    for w in ins_words:
                        if w in emb.vocab:
                            target_vectors.append(emb.get_vector(w))
                        else:
                            target_vectors.append(emb.get_vector('<oov>'))
                    target_vectors = np.array(target_vectors)

                    # Add language cost to the accumulator
                    loss = 0
                    for pos, target_vector in enumerate(target_vectors):
                        curr_vector = produced_vectors[pos]
                        # if np.array_equal(target_vector, eos_vector):
                        #    break
                        # if np.array_equal(curr_vector, eos_vector):
                        #    break
                        ## TODO: Try different loss functions
                        ## L2 loss
                        # loss += torch.mean((curr_vector - torch.from_numpy(target_vector).cuda()) ** 2)
                        ## Cosine loss
                        loss += torch.nn.functional.cosine_similarity(curr_vector.squeeze(), torch.from_numpy(target_vector).cuda(), dim=0)

                    losses[idx] = loss

                language_loss = torch.min(losses)

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
            (policy_loss + 0.5 * value_loss + 0.1 * language_loss).backward()
        else:
            (policy_loss + 0.5 * value_loss).backward()

        # (policy_loss + 0.5 * value_loss).backward()
        print("****************")
        print(policy_loss)
        print(value_loss)
        if args.use_language:
            print(language_loss)

        print("****************")
        # """
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()

        # Clean agent observations
        player.clear_actions()
