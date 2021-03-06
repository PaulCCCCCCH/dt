import numpy as np
from scipy import spatial
from utils import read_pong_instructions
from params import args

def is_alpha(word):
    try:
        return word.encode('ascii').isalpha()
    except:
        return False


class Embedding:
    def __init__(self, emb_path=None, specific_vocab=None):
        self.emb_dim = args.emb_dim
        self.emb_to_load = args.emb_to_load
        self.vocab = []
        self.emb_mat = None
        self._specific_vocab = specific_vocab
        self._specific_vocab_set = set(specific_vocab)
        self._word_indices = dict()
        self._load_embedding(emb_path)

    def _load_embedding(self, emb_path):
        count = 0
        emb = []

        if emb_path is not None:
            with open(emb_path, 'r') as f:
                for line in f:
                    s = line.split()
                    if count >= self.emb_to_load and self._specific_vocab is None:
                        break

                    if count >= self.emb_to_load:
                        if s[0] not in self._specific_vocab_set:
                            continue

                    if is_alpha(s[0]):
                        self.vocab.append(s[0])
                        self._word_indices[s[0]] = count
                        emb.append(np.asarray(s[1:], dtype=np.float32))
                        count += 1
        else:
            assert self._specific_vocab is not None
            for v in self._specific_vocab:
                self.vocab.append(v)
                self._word_indices[v] = count
                emb.append(np.random.randn(self.emb_dim))
                count += 1

        for index, word in enumerate(self.vocab):
            self._word_indices[word] = index

        self.emb_mat = np.array(emb)

        # Append special words to the embedding model
        direction = np.zeros(args.emb_dim)
        direction[0] = 1
        self.add("<eos>", direction)  # ignore the warning here
        direction[0] = 0
        direction[1] = 1
        self.add("<pad>", direction)
        direction[1] = 0
        direction[2] = 1
        self.add("<oov>", direction)
        direction[2] = 0
        direction[3] = 1
        self.add("<sos>", direction)

    @property
    def index2entity(self):
        return self.vocab

    def get_vector(self, word):
        return self.emb_mat[self._word_indices[word]]

    def add(self, word, init=None):
        self._word_indices[word] = len(self.vocab)
        self.vocab.append(word)
        if init is None:
            # init = np.random.randn(self.emb_dim)
            init = np.zeros(self.emb_dim)
        init = np.array(init)
        self.emb_mat = np.vstack((self.emb_mat, init))

    def distances(self, word_or_vec):
        if type(word_or_vec) == str:
            word_vec = self.get_vector(word_or_vec)
        else:
            word_vec = word_or_vec
        dists = []
        for w in self.vocab:
            target_vec = self.get_vector(w)
            # dist = np.sum(np.square(target_vec - word_vec))
            dist = spatial.distance.cosine(target_vec, word_vec)
            dists.append(dist)
        return dists

    def get_index(self, word):
        return self._word_indices[word]


if __name__ == '__main__':
    # Test
    path = '../emb/glove_twitter_25d_changed.txt'
    instructions, specific_vocab, _ = read_pong_instructions("./data/pong.txt")
    emb = Embedding(emb_path=path, specific_vocab=specific_vocab)
    emb.add("<eos>")
    emb.add("<pad>")
    emb.add("<oov>")

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
