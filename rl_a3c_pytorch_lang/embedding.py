import numpy as np
from utils import read_pong_instructions

def is_alpha(word):
    try:
        return word.encode('ascii').isalpha()
    except:
        return False


class Embedding:
    def __init__(self, emb_path, specific_vocab=None):
        self.emb_dim = 25
        self.emb_to_load = 1000
        self.vocab = []
        self.emb_mat = None
        self._specific_vocab = specific_vocab
        self._word_indices = dict()
        self._load_embedding(emb_path)

    def _load_embedding(self, emb_path):
        count = 0
        emb = []

        with open(emb_path, 'r') as f:
            for line in f:
                s = line.split()
                if count >= self.emb_to_load and self._specific_vocab is None:
                    break

                if count >= self.emb_to_load:
                    if s[0] not in self._specific_vocab:
                        continue

                if is_alpha(s[0]):
                    self.vocab.append(s[0])
                    self._word_indices[s[0]] = count
                    emb.append(np.asarray(s[1:], dtype=np.float32))
                    count += 1

        for index, word in enumerate(self.vocab):
            self._word_indices[word] = index

        self.emb_mat = np.array(emb)

    @property
    def index2entity(self):
        return self.vocab

    def get_vector(self, word):
        return self.emb_mat[self._word_indices[word]]

    def add_word(self, word, init=None):
        self._word_indices[word] = len(self.vocab)
        self.vocab.append(word)
        if init is None:
            # init = np.random.randn(self.emb_dim)
            init = np.zeros(self.emb_dim)
        self.emb_mat = np.vstack((self.emb_mat, init))

    def distances(self, word_or_vec):
        if type(word_or_vec) == str:
            word_vec = self.get_vector(word_or_vec)
        else:
            word_vec = word_or_vec
        dists = []
        for w in self.vocab:
            target_vec = self.get_vector(w)
            dist = np.sum(np.square(target_vec - word_vec))
            dists.append(dist)
        return dists


if __name__ == '__main__':
    # Test
    path = '../emb/glove_twitter_25d_changed.txt'
    instructions, specific_vocab = read_pong_instructions("./data/pong.txt")
    emb = Embedding(emb_path=path, specific_vocab=specific_vocab)
    emb.add_word("<eos>")
    emb.add_word("<pad>")
    emb.add_word("<oov>")

    print(len(emb.vocab))
    print(len(emb.emb_mat))
    word = "<eos>"
    print(emb.get_vector(word))
    print(emb.emb_mat[emb._word_indices[word]])
    word = "word"
    print(emb.get_vector(word))
    print(emb.emb_mat[emb._word_indices[word]])

    word = "bad"
    print(np.argmin(emb.distances(word)))
    print(emb._word_indices[word])
