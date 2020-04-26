from __future__ import division
import numpy as np
import torch
import json
import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        """
        print("***************")
        print(shared_param)
        print(shared_param.grad)
        print(shared_param._grad)
        print("***************")
        """
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


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


def find_closest(emb, word, return_vector=False, return_word=False):

    distances = emb.distances(word)
    closest_word = emb.index2entity[np.argmin(distances)]
    closest_vector = emb.get_vector(closest_word)

    if return_vector and return_word:
        return (closest_vector, closest_word)
    elif return_word:
        return closest_word
    elif return_vector:
        return closest_vector


def read_pong_instructions(path):
    instruction_sets = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    bi_grams = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    vocab_set = set()
    with open(path, "r") as f:

        lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line[:-1].split('%%%')
            label = int(parts[0])
            sentence = parts[1]
            sentence += " <eos>"
            while len(sentence.split()) < 10:
                sentence += " <pad>"

            # Adding words in the instructions to the vocabulary
            for w in sentence.split():
                vocab_set.add(w)

            # Building bi-gram model
            bi_gram = bi_grams[label]
            sent_words = sentence.split()
            for pos in range(len(sent_words) - 1):

                curr_word = sent_words[pos]
                next_word = sent_words[pos + 1]

                if curr_word not in bi_gram:
                    bi_gram[curr_word] = {}

                if next_word not in bi_gram[curr_word]:
                    bi_gram[curr_word][next_word] = 1
                else:
                    bi_gram[curr_word][next_word] += 1
            if label == 2:
                bi_grams[4] = bi_gram
            elif label == 3:
                bi_grams[5] = bi_gram
            elif label == 0:
                bi_grams[1] = bi_gram


            # Classifying instructions
            if label == 2:
                instruction_sets[2].append(sentence)
                instruction_sets[4].append(sentence)
            elif label == 3:
                instruction_sets[3].append(sentence)
                instruction_sets[5].append(sentence)
            elif label == 0:
                instruction_sets[0].append(sentence)
                instruction_sets[1].append(sentence)

    new_bi_grams = {}
    for a in range(len(bi_grams)):
        new_bi_grams[a] = {}
        bi_gram = bi_grams[a]
        for w1 in bi_gram.keys():
            new_bi_grams[a][w1] = {}
            count = float(sum(bi_gram[w1].values()))
            for w2 in bi_gram[w1].keys():
                new_bi_grams[a][w1][w2] = bi_gram[w1][w2] / count

    return instruction_sets, vocab_set, new_bi_grams


def read_mr_instructions(path):
    instruction_sets = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    bi_grams = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    vocab_set = set()
    with open(path, "r") as f:

        lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line[:-1].split('\t')
            sentence = parts[1]
            sentence += " <eos>"
            while len(sentence.split()) < 10:
                sentence += " <pad>"

            # Adding words in the instructions to the vocabulary
            for w in sentence.split():
                vocab_set.add(w)

            # Building bi-gram model
            bi_gram = bi_grams[label]
            sent_words = sentence.split()
            for pos in range(len(sent_words) - 1):

                curr_word = sent_words[pos]
                next_word = sent_words[pos + 1]

                if curr_word not in bi_gram:
                    bi_gram[curr_word] = {}

                if next_word not in bi_gram[curr_word]:
                    bi_gram[curr_word][next_word] = 1
                else:
                    bi_gram[curr_word][next_word] += 1
            if label == 2:
                bi_grams[4] = bi_gram
            elif label == 3:
                bi_grams[5] = bi_gram
            elif label == 0:
                bi_grams[1] = bi_gram


            # Classifying instructions
            if label == 2:
                instruction_sets[2].append(sentence)
                instruction_sets[4].append(sentence)
            elif label == 3:
                instruction_sets[3].append(sentence)
                instruction_sets[5].append(sentence)
            elif label == 0:
                instruction_sets[0].append(sentence)
                instruction_sets[1].append(sentence)

    new_bi_grams = {}
    for a in range(len(bi_grams)):
        new_bi_grams[a] = {}
        bi_gram = bi_grams[a]
        for w1 in bi_gram.keys():
            new_bi_grams[a][w1] = {}
            count = float(sum(bi_gram[w1].values()))
            for w2 in bi_gram[w1].keys():
                new_bi_grams[a][w1][w2] = bi_gram[w1][w2] / count

    return instruction_sets, vocab_set, new_bi_grams


if __name__ == '__main__':
    ins_set, vocab_set, bi_grams = read_pong_instructions("./data/pong.txt")
    print(bi_grams)
