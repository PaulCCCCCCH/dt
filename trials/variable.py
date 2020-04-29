from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from utils import read_mr_instructions, to_one_hot
from embedding import Embedding

class MyModel(torch.nn.Module):
    def __init__(self, emb):
        super(MyModel, self).__init__()
        self.emb_mat = torch.nn.Parameter(torch.from_numpy(emb.emb_mat).to(dtype=torch.float), requires_grad=True)
        # self.emb_mat = torch.nn.Parameter(torch.from_numpy(emb_mat)).float().cuda()

        # self.out_layer = nn.Linear(emb_mat.shape[1], 3).cuda()
        # self.lstm = nn.LSTMCell(self.emb_mat.shape[1], 3).cuda()
        self.lstm = nn.LSTMCell(self.emb_mat.shape[1], 25).cuda()

        self.train()

    def forward(self, inputs):
        """
        out = []
        for input in inputs:
            input = torch.from_numpy(input).float().cuda()
            mult = torch.mm(input.unsqueeze(0), self.emb_mat)
            out.append(self.out_layer(mult))


        return out

        """
        # inputs = torch.from_numpy(inputs).float().cuda()
        embs = torch.mm(inputs, self.emb_mat)
        out = []
        hx_enc = torch.zeros(1, self.emb_mat.shape[1]).cuda()
        cx_enc = torch.zeros(1, self.emb_mat.shape[1]).cuda()

        for emb in embs:

            hx_enc, cx_enc = self.lstm(emb.unsqueeze(0), (hx_enc, cx_enc))
            out.append(hx_enc)
            # out.append(self.out_layer(emb.unsqueeze(0)))

        return out


class Dan(nn.Module):
    def __init__(self):
        super(Dan, self).__init__()
        self.alpha = nn.Parameter(torch.from_numpy(np.array([[1.,2,3]])))


if __name__ == "__main__":

    data_path = "../montezuma_data/annotations.txt"
    emb_path = "../emb/glove_twitter_25d_changed.txt"
    instructions, vocab = read_mr_instructions(data_path)
    emb = Embedding(emb_path, vocab)
    vocab_size = len(emb.vocab)

    model = MyModel(emb).cuda()
    # print(list(model.parameters()))

    sample = instructions[500]
    input_one_hot = []
    input_one_hot.append(to_one_hot(emb.get_index("<sos>"), vocab_size))
    for w in sample:
        if w not in emb.vocab:
            w = "<oov>"
        one_hot = to_one_hot(emb.get_index(w), vocab_size)
        input_one_hot.append(one_hot)
        if w == "<eos>":
            break
    # Ignore <eos> for input
    input_one_hot_no_eos = torch.from_numpy(np.array(input_one_hot[:-1])).to(dtype=torch.float).cuda()
    input_one_hot = torch.from_numpy(np.array(input_one_hot)).to(dtype=torch.float).cuda()
    trial_input = input_one_hot

    """
    trial_input = np.array([[0.4, 0.6], [0.7, 0.8], [0.7, 0.8], [0.7, 0.8], [0.7, 0.8], [0.7, 0.8]])
    """
    trial_output = model(trial_input)
    # print(trial_output)
    target = torch.tensor(1).cuda()
    loss = 0
    for i in range(len(trial_output)):
        loss += torch.nn.functional.cross_entropy(trial_output[i], target.unsqueeze(0))
    loss.backward()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    print(model.emb_mat.grad)

