from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init, find_closest
from params import args

USE_LANGUAGE = args.use_language
EMB_DIM = args.emb_dim


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space, emb):
        self.alpha = 0.2  # Weight for language component
        self.emb = emb

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

            # LSTM for encoding state into language for actor
            self.lstm_enc = nn.LSTMCell(EMB_DIM, EMB_DIM)

            # LSTM for decoding state
            self.lstm_dec = nn.LSTMCell(EMB_DIM, EMB_DIM)

            # Actor (Logits from decoder -> Action logits)
            self.actor_lang_linear = nn.Linear(EMB_DIM, num_outputs)

            # Actor (State -> State repr for language model)
            self.actor_lang_prep = nn.Linear(args.lstm_size, EMB_DIM)

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

    def forward(self, inputs):
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
            actor_lang_input = self.actor_lang_prep(x)

            # Encoder
            hx_enc = torch.zeros_like(actor_lang_input)
            cx_enc = torch.zeros_like(actor_lang_input)
            encoder_output_vectors = []
            inp = actor_lang_input
            for _ in range(10): # TODO: 10 is a hyper parameter (seq len)
                hx_enc, cx_enc = self.lstm_enc(inp, (hx_enc, cx_enc))
                inp = hx_enc
                encoder_output_vectors.append(hx_enc)

            # Decoder
            hx_dec = torch.zeros_like(hx_enc)
            cx_dec = torch.zeros_like(hx_enc)
            for i in range(10):
                hx_dec, cx_dec = self.lstm_dec(encoder_output_vectors[i], (hx_dec, cx_dec))

            actor_lang = self.actor_lang_linear(hx_dec)
        else:
            actor_lang = 0
            encoder_output_vectors = None

        actor_out = self.alpha * actor_lang + (1 - self.alpha) * actor_fc

        return encoder_output_vectors, critic_out, actor_out, (hx, cx)
