from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class LangNN(torch.nn.Module):
    def __init__(self, action_space):

        super(LangNN, self).__init__()

        num_outputs = action_space

        # Actor (State -> Action Probabilities)
        self.actor_lang_linear = nn.Linear(25, num_outputs)

        # LSTM for encoding state into language for actor
        self.lstm_enc = nn.LSTMCell(25, 25)

        # LSTM for decoding state
        self.lstm_dec = nn.LSTMCell(25, 25)

        self.apply(weights_init)
        self.actor_lang_linear.weight.data = norm_col_init(
            self.actor_lang_linear.weight.data, 0.01)
        self.actor_lang_linear.bias.data.fill_(0)

        self.lstm_enc.bias_ih.data.fill_(0)
        self.lstm_enc.bias_hh.data.fill_(0)

        self.lstm_dec.bias_ih.data.fill_(0)
        self.lstm_dec.bias_hh.data.fill_(0)

        self.train()

    def forward(self, state_repr_input):

        # Encoder
        hx_enc = torch.zeros_like(state_repr_input)
        cx_enc = torch.zeros_like(state_repr_input)
        encoder_output_vectors = []
        inp = state_repr_input
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


        return actor_lang, encoder_output_vectors


class TrialNN(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        self.alpha = 0.2  # Weight for language component

        super(TrialNN, self).__init__()
        self.lang_model = LangNN(action_space)


        for param in self.lang_model.parameters():
            param.requires_grad = True

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        # 1024 = 64 * 64 / 4

        self.flatten = nn.Linear(1024, 25)

        self.lstm = nn.LSTMCell(25, 25)
        num_outputs = action_space

        # Critic (State -> Value)
        self.critic_linear = nn.Linear(25, 1)

        # Actor (State -> Action Probabilities)
        self.actor_linear = nn.Linear(25, num_outputs)

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

        actor_lang = self.lang_model(x)
        actor_out, encoder_output_vectors = self.alpha * actor_lang + (1 - self.alpha) * actor_fc

        return encoder_output_vectors, critic_out, actor_out, (hx, cx)


if __name__ == '__main__':
    action_space = 10
    create = True
    model = TrialNN(32, action_space)

    if create:
        print(model)
        params = dict(model.named_parameters())
        print(params)
        print('========================')
        print('========================')
        print('========================')
        print('========================')
        print(model.state_dict())
        torch.save(model.state_dict(), './temp.dat')
    else:
        print(model)
        params = dict(model.named_parameters())
        print(params)
        print('========================')
        print('========================')
        print('========================')
        print('========================')
        saved_state = torch.load('./temp.dat', map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state)
        print(model.state_dict())
