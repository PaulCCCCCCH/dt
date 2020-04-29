from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import Embedding
from utils import norm_col_init, weights_init, find_closest, read_mr_instructions, to_one_hot, read_pong_instructions
from params import args
import random
import numpy as np
import torch.optim as optim
import os
import pickle
import time

EMB_DIM = args.emb_dim
LSTM_SIZE = args.lstm_size
MAX_ITER = None

class LangModel(torch.nn.Module):
    def __init__(self, emb):
        super(LangModel, self).__init__()

        self.alpha = 0.5  # Weight for language component
        self.emb = emb
        self.emb_mat = torch.nn.Parameter(torch.from_numpy(emb.emb_mat).float().cuda(), requires_grad=True)

        # self.emb_mat = torch.from_numpy(emb.emb_mat).float().cuda()

        self.dropout = nn.Dropout(p=0.8)

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

    def forward(self, inputs):
        """
        Expect inputs to be one-hot tensor of shape [SeqLength, VocabSize]
        """
        hx_enc = torch.zeros(1, LSTM_SIZE).cuda()
        cx_enc = torch.zeros(1, LSTM_SIZE).cuda()
        encoder_output_logits = []

        if not args.eval:
            assert inputs is not None
            # Shape [SeqLength, EmbeddingDim]
            x_emb = torch.mm(inputs, self.emb_mat)
            for i in range(x_emb.shape[0]):
                hx_enc, cx_enc = self.lstm_enc(x_emb[i].unsqueeze(0), (hx_enc, cx_enc))
                hx_enc = self.dropout(hx_enc)
                # Shape [1, vocab_size]
                lang_logit = self.linear_lang_gen(hx_enc)
                encoder_output_logits.append(lang_logit)
        else:
            # word = None
            # input_emb = self.emb.get_vector("<sos>")
            # print("Please input first word")
            # word = input()
            word = "<sos>"
            while word != '<eos>':
                input_emb = torch.from_numpy(self.emb.get_vector(word)).to(dtype=torch.float).unsqueeze(0).cuda()
                hx_enc, cx_enc = self.lstm_enc(input_emb, (hx_enc, cx_enc))
                # Shape [1, vocab_size]
                lang_logit = self.linear_lang_gen(hx_enc)
                encoder_output_logits.append(lang_logit)

                # Eval method 1: argmax selection
                """
                next_idx = torch.argmax(lang_logit)
                next_word = self.emb.vocab[next_idx]
                word = next_word
                print(next_word)
                """

                # Eval method 2: random sampling
                word_prob = torch.nn.functional.softmax(lang_logit.detach().cpu(), dim=1)[0]
                next_idx = np.random.choice(len(word_prob), 1, p=word_prob.numpy())[0]

                next_word = self.emb.vocab[next_idx]
                word = next_word
                print(next_word)
                # time.sleep(0.1)



        return encoder_output_logits


if __name__ == "__main__":

    model_name = "language_model_50d"

    save_dir = "./pre_trained_lang_model"
    save_path = os.path.join(save_dir, model_name + '.dat')
    emb_pickle_path = os.path.join(save_dir, model_name + '_emb.pkl')

    mz_data_path = "./data/annotations.txt"
    pong_data_path = "./data/pong.txt"
    # emb_path = "../emb/glove_twitter_25d_changed.txt"
    instructions, vocab_mr = read_mr_instructions(mz_data_path)
    _, vocab_pong, _ = read_pong_instructions(pong_data_path)
    vocab = vocab_mr.union(vocab_pong)

    if args.use_ckpt or args.eval:
        with open(emb_pickle_path, "rb") as f:
            emb = pickle.load(f)
    else:
        emb = Embedding(emb_path=None, specific_vocab=vocab)

    vocab_size = len(emb.vocab)

    print("Vocab size is:")
    print(vocab_size)
    print("Emb shape is")
    print(emb.emb_mat.shape)

    model = LangModel(emb).cuda()
    optimizer = optim.Adam(model.parameters())

    if args.use_ckpt:
        state_to_load = torch.load(save_path,
            map_location=lambda storage, loc: storage)
        model.load_state_dict(state_to_load)

    # Train or eval
    if not args.eval:
        # while True:
        if MAX_ITER is None:
            MAX_ITER = 999999999
        for step in range(MAX_ITER):
            # Sample and prepare input
            sample_idx = random.randrange(0, len(instructions))
            # sample: [str]
            sample = instructions[sample_idx]
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

            # Predict
            # logits: list of [1, vocab_size] tensors
            logits = model(input_one_hot_no_eos)

            # Calculate loss
            loss = 0

            for i, logit in enumerate(logits):
                target_class = torch.argmax(input_one_hot[i + 1]).cuda()
                loss += torch.nn.functional.cross_entropy(logit, target_class.unsqueeze(0))
            loss /= len(logits)

            loss.backward()
            optimizer.step()
            model.zero_grad()

            if step % 100 == 0:
                if step > 0:
                    sentence = [emb.vocab[torch.argmax(logit)] for logit in logits]
                    sentence = " ".join(sentence)

                    print("Loss at step {}: {}".format(step, loss))
                    print("Predicted sentence at step {}: {}".format(step, sentence))
                    print("Target sentence at step {}: {}".format(step, sample))
                    # Save model parameters
                    state_to_save = model.state_dict()
                    torch.save(state_to_save, save_path)

                    # Save embeddings
                    emb.emb_mat = model.emb_mat.data.cpu().numpy()
                    with open(emb_pickle_path, "wb") as f:
                        pickle.dump(emb, f)


    else:
        state_to_load = torch.load(save_path,
            map_location=lambda storage, loc: storage)

        model.load_state_dict(state_to_load)
        model.eval()

        inputs = torch.randn(1, vocab_size)
        logits = model(inputs)
        sent = [emb.vocab[torch.argmax(logit)] for logit in logits]

        print(sent)
