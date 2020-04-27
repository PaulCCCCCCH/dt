from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import find_closest


class Agent(object):
    def __init__(self, model, env, args, state, emb):
        self.emb = emb
        self.produced_vectors = []
        self.produced_logits = []
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
        self.action_logits = []

        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
    # Run a step and save observations in the lists. These will be
    # fetched for updating weights in the future.
    def action_train(self):
        (vectors, logits), value, logit, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.action_logits.append(logit)
        self.entropies.append(entropy)
        self.produced_vectors.append(vectors)
        self.produced_logits.append(logits)
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

    def action_test(self, manual_language_input=None):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, self.args.lstm_size).cuda())
                        self.hx = Variable(
                            torch.zeros(1, self.args.lstm_size).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, self.args.lstm_size))
                    self.hx = Variable(torch.zeros(1, self.args.lstm_size))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)

            # TODO: This is only for comparison
            (vectors_, logits_), value_, logit_, (_, _) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))


            (vectors, logits), value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)), manual_language_input)




        # TODO: This is only for comparison
        prob_ = F.softmax(logit_, dim=1)
        action_ = prob_.max(1)[1].data.cpu().numpy()

        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1

        if self.args.use_language:
            if self.args.render:
                if self.eps_len % 100 == 0:
                # if self.eps_len % 1 == 0:

                    sent = []
                    # Deterministic language generation using output embedding matching
                    """
                    for vec in vectors:
                        if vec.is_cuda:
                            vec = vec.cpu()
                        word = find_closest(self.emb, vec.squeeze().numpy(), return_word=True)
                        sent.append(word)
                    """

                    if self.args.rand_gen:
                        # Probabilistic language generation using logits
                        for logit in logits:
                            logit = torch.nn.functional.softmax(logit)
                            idx = torch.multinomial(logit, 1)
                            word = self.emb.vocab[idx]
                            sent.append(word)

                    else:
                        # Deterministic language generation using logits

                        sent = []
                        for logit in logits:
                            idx = torch.argmax(logit)
                            word = self.emb.vocab[idx]
                            sent.append(word)

                    print(str(action), str(action_) + " ".join(sent), len(sent))


        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.produced_vectors = []
        self.produced_logits = []
        self.action_logits = []
        return self
