from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import Embedding
from utils import norm_col_init, weights_init, find_closest, read_mr_instructions, to_one_hot
from params import args
import random
import numpy as np
import torch.optim as optim
import os

EMB_DIM = 25
LSTM_SIZE = 100
MAX_ITER = 10

class LangModel(torch.nn.Module):
    def __init__(self, emb):
        super(LangModel, self).__init__()

        self.alpha = 0.5  # Weight for language component
        self.emb = emb
        self.emb_mat = torch.nn.Parameter(torch.from_numpy(emb.emb_mat).to(dtype=torch.float), requires_grad=True)
        # self.emb_mat = torch.from_numpy(emb.emb_mat).float().cuda()

        # LSTM for encoding state into language for actor
        self.lstm_enc = nn.LSTMCell(EMB_DIM, LSTM_SIZE)

        # LSTM for encoding state into language for actor
        self.linear_lang_gen = nn.Linear(LSTM_SIZE, len(emb.vocab))

        # Initialisation
        self.apply(weights_init)

        # Initialising actor language layer weights
        self.lstm_enc.bias_ih.data.fill_(0)
        self.lstm_enc.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs, is_train=True):
        """
        Expect inputs to be one-hot tensor of shape [SeqLength, VocabSize]
        """
        encoder_output_logits = []

        # Shape [SeqLength, EmbeddingDim]
        x_emb = torch.mm(inputs, self.emb_mat)

        hx_enc = torch.zeros(1, LSTM_SIZE).cuda()
        cx_enc = torch.zeros(1, LSTM_SIZE).cuda()

        if is_train:
            for i in range(x_emb.shape[0]):
                hx_enc, cx_enc = self.lstm_enc(x_emb[i].unsqueeze(0), (hx_enc, cx_enc))
                # Shape [1, vocab_size]
                lang_logit = self.linear_lang_gen(hx_enc)
                encoder_output_logits.append(lang_logit)
        else:
            word = None
            input_emb = self.emb.get_vector("<sos>")
            while word != '<eos>':
                hx_enc, cx_enc = self.lstm_enc(input_emb, (hx_enc, cx_enc))
                # Shape [1, vocab_size]
                lang_logit = self.linear_lang_gen(hx_enc)
                encoder_output_logits.append(lang_logit)

                next_idx = torch.argmax(lang_logit)
                next_word = self.emb.vocab[next_idx]
                word = next_word


        return encoder_output_logits


if __name__ == "__main__":

    model_name = "trial_lm"

    save_dir = "./pre_trained_lang_model"
    save_path = os.path.join(save_dir, model_name)
    data_path = "../montezuma_data/annotations.txt"
    emb_path = "../emb/glove_twitter_25d_changed.txt"
    instructions, vocab = read_mr_instructions(data_path)
    emb = Embedding(emb_path, vocab)
    vocab_size = len(emb.vocab)

    model = LangModel(emb).cuda()
    optimizer = optim.Adam(model.parameters())


    # while True:
    for step in range(MAX_ITER):
        # Sample and prepare input
        sample_idx = random.randrange(0, len(instructions))
        sample = instructions[sample_idx]
        input_one_hot = []
        for w in sample:
            one_hot = to_one_hot(emb.get_index(w), vocab_size)
            input_one_hot.append(one_hot)
            if w == "<eos>":
                break
        input_one_hot = torch.from_numpy(np.array(input_one_hot)).to(dtype=torch.float).cuda()

        # Predict
        logits = model(input_one_hot)

        # Calculate loss
        loss = 0

        for logit in logits:
            target_class = torch.argmax(logit).cuda()
            loss += torch.nn.functional.cross_entropy(logit, target_class.unsqueeze(0))
        loss /= len(logits)

        loss.backward()
        optimizer.step()
        model.zero_grad()

        for i in range(len(instructions)):
            sent = instructions[i]
            if 'slighly' in sent:
                print("hhhhhhhhhhhhhhhhh")

        if step % 100 == 0:
            print("Loss at step {}: {}".format(step, loss))

            state_to_save = model.state_dict()
            torch.save(state_to_save, save_path)

