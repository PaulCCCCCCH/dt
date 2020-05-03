from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from embedding import Embedding
from utils import norm_col_init, weights_init, find_closest
from params import args

USE_LANGUAGE = args.use_language
EMB_DIM = args.emb_dim


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space, emb):
        super(A3Clstm, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        # 1024 = 64 * 64 / 4

        self.flatten = nn.Linear(1024, args.lstm_size)

        self.lstm = nn.LSTMCell(args.lstm_size, args.lstm_size)
        num_outputs = action_space.n

        if USE_LANGUAGE:
            self.alpha = 0.5  # Weight for language component
            self.emb = emb
            self.emb_mat = torch.nn.Parameter(torch.from_numpy(emb.emb_mat).to(dtype=self.lstm.bias_hh.dtype), requires_grad=True)
            # self.emb_mat = torch.from_numpy(emb.emb_mat).float().cuda()

            # LSTM for encoding state into language for actor
            self.lstm_enc = nn.LSTMCell(args.lstm_size, args.lstm_size)

            # LSTM for encoding state into language for actor
            self.linear_lang_gen = nn.Linear(args.lstm_size, len(emb.vocab))

            # LSTM for decoding state
            self.lstm_dec = nn.LSTMCell(EMB_DIM, args.lstm_size)

            # Actor (Logits from decoder -> Action logits)
            self.actor_lang_linear = nn.Linear(args.lstm_size, num_outputs)

            # Actor (State -> State repr for language model)
            self.actor_lang_prep = nn.Linear(args.lstm_size, args.lstm_size)

        # Critic (State -> Value)
        self.critic_linear = nn.Linear(args.lstm_size, 1)

        # Actor (State -> Action logits)
        self.actor_linear = nn.Linear(args.lstm_size, num_outputs)



        # Define the language model

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # Initialising actor language layer weights
        if USE_LANGUAGE:
            self.actor_lang_linear.weight.data = norm_col_init(
                self.actor_lang_linear.weight.data, 0.01)
            self.actor_lang_linear.bias.data.fill_(0)
            self.lstm_enc.bias_ih.data.fill_(0)
            self.lstm_enc.bias_hh.data.fill_(0)

            self.lstm_dec.bias_ih.data.fill_(0)
            self.lstm_dec.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs, manual_language_input=None):
        inputs, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = self.flatten(x)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx
        critic_out = self.critic_linear(x)
        actor_fc = self.actor_linear(x)

        if USE_LANGUAGE:
            # actor_lang_input = torch.softmax(self.actor_lang_prep(x), dim=0)
            actor_lang_input = self.actor_lang_prep(x)

            # Encoder
            """
            hx_enc = torch.zeros_like(actor_lang_input)
            cx_enc = torch.zeros_like(actor_lang_input)
            """
            hx_enc = torch.zeros_like(actor_lang_input)
            cx_enc = actor_lang_input

            encoder_output_vectors = []
            encoder_output_logits = []
            inp = actor_lang_input

            # No manual input
            if manual_language_input is None:
                for _ in range(10): # TODO: 10 is a hyper parameter (seq len)
                    hx_enc, cx_enc = self.lstm_enc(inp, (hx_enc, cx_enc))
                    lang_logit = self.linear_lang_gen(hx_enc)
                    inp = hx_enc
                    encoder_output_logits.append(lang_logit)
                    encoder_output_vectors.append(torch.mm(torch.softmax(lang_logit, 1), self.emb_mat))

            # Manual input
            else:
                # Expect manual language input type list[str] with length 10 and <eos> ending
                encoder_output_vectors = [torch.from_numpy(self.emb.get_vector(w)).to(dtype=inp.dtype).unsqueeze(0).cuda() for w in manual_language_input]
                for w in manual_language_input:
                    lang_logit = torch.zeros(len(self.emb.vocab)).cuda()
                    word_idx = self.emb.get_index(w)
                    lang_logit[word_idx] = 1
                    encoder_output_logits.append(lang_logit.unsqueeze(0))

            # Decoder
            """
            Since hx_enc will not be available during testing (instruction is given), use all
            zero here
            """
            hx_dec = torch.zeros(1, args.lstm_size).cuda()
            cx_dec = torch.zeros(1, args.lstm_size).cuda()
            for i in range(10):
                hx_dec, cx_dec = self.lstm_dec(encoder_output_vectors[i], (hx_dec, cx_dec))

            actor_lang = self.actor_lang_linear(hx_dec)
        else:
            actor_lang = 0
            encoder_output_vectors = None
            encoder_output_logits = None

        if not args.manual_control:
            r = random.random()
            if r < 0.33:
                self.alpha = 1
            elif r < 0.67:
                self.alpha = 0.5
            else:
                self.alpha = 0

        actor_out = self.alpha * actor_lang + (1 - self.alpha) * actor_fc

        return (encoder_output_vectors, encoder_output_logits), critic_out, actor_out, (hx, cx)

