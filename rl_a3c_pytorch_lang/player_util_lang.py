from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import find_closest


class Agent(object):
    def __init__(self, model, env, args, state, emb):
        self.emb = emb
        self.produced_vectors = []
        self.target_vectors = []

        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []

        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1

    # Run a step and save observations in the lists. These will be
    # fetched for updating weights in the future.
    def action_train(self):
        vectors, value, logit, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        self.produced_vectors.append(vectors)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        # Step
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, 25).cuda())
                        self.hx = Variable(
                            torch.zeros(1, 25).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 25))
                    self.hx = Variable(torch.zeros(1, 25))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            vectors, value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))

        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1

        if self.args.render:
            sent = []
            for vec in vectors:
                word = find_closest(self.emb, vec.squeeze(), return_word=True)
                sent.append(word)
            print(str(action) + " ".join(sent), len(sent))
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.produced_vectors = []
        return self
