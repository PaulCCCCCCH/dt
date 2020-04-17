from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Linear(1024, 100)

        self.lstm_enc = nn.LSTMCell(100, 100)
        num_outputs = action_space.n

        self.lstm_actor_dec = nn.LSTMCell(100, 100)
        self.lstm_critic_dec = nn.LSTMCell(100, 100)

        self.critic_linear = nn.Linear(100, 1)
        self.actor_linear = nn.Linear(100, num_outputs)

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

        self.lstm_enc.bias_hh.data.fill_(0)
        self.lstm_enc.bias_ih.data.fill_(0)
        self.lstm_actor_dec.bias_hh.data.fill_(0)
        self.lstm_actor_dec.bias_ih.data.fill_(0)
        self.lstm_critic_dec.bias_hh.data.fill_(0)
        self.lstm_critic_dec.bias_ih.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (prev_hx, prev_cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        inp = self.flatten(x)
        hx = prev_hx
        cx = prev_cx

        # Encoding state information into language
        hxs = []
        for _ in range(10):
            hx, cx = self.lstm_enc(inp, (hx, cx))
            hxs.append(hx)
            inp = hx

        # Saving new LSTM states for next iteration
        new_hx = hx
        new_cx = cx

        # Decoding natural language vectors into information
        hx = prev_hx
        cx = prev_cx
        hx2 = hx
        cx2 = cx

        for i in range(10):
            hx, cx = self.lstm_actor_dec(hxs[i], (hx, cx))
            hx2, cx2 = self.lstm_critic_dec(hxs[i], (hx2, cx2))

        return self.critic_linear(hx2), self.actor_linear(hx), (new_hx, new_cx)
