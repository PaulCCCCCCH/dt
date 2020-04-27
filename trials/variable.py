from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MyModel(torch.nn.Module):
    def __init__(self, emb_mat):
        super(MyModel, self).__init__()
        self.emb_mat = torch.nn.Parameter(torch.from_numpy(emb_mat).float().cuda())

        self.out_layer = nn.Linear(emb_mat.shape[1], 3).cuda()

        self.train()

    def forward(self, inputs):
        inputs = torch.from_numpy(inputs).float().cuda()
        mult = torch.mm(inputs, self.emb_mat)
        out = self.out_layer(mult)

        return out


class Dan(nn.Module):
    def __init__(self):
        super(Dan, self).__init__()
        self.alpha = nn.Parameter(torch.from_numpy(np.array([[1.,2,3]])))


if __name__ == "__main__":
    m = np.array([[0.1, 0.2, 0.3], [0.04, 0.05, 0.06]])
    model = MyModel(m)
    print(list(model.parameters()))
    trial_input = np.array([[0.4, 0.6]])
    trial_output = model(trial_input)
    target = torch.tensor(1).cuda()
    loss = torch.nn.functional.cross_entropy(trial_output, target.unsqueeze(0))
    loss.backward()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    print(model.emb_mat)
    print(model.emb_mat.grad)

