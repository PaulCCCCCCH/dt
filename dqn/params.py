import os
from collections import namedtuple

EnvInfo = namedtuple("EnvInfo", ["name", "offset_height", "offset_width", "target_height", "target_width"])

envs = {0: EnvInfo('Breakout-v0', 34, 0, 160, 160),
        1: EnvInfo('MontezumaRevenge-v0', 34, 0, 160, 160),
        2: EnvInfo('Pong-v0', 34, 0, 160, 160),
        3: EnvInfo('MsPacman-v0', 0, 0, 160, 160),
        4: EnvInfo('SpaceInvaders-v0', 0, 0, 210, 160)}

class Config:
    def __init__(self):

        # Modes
        self.is_train = True
        self.is_debug = False
        self.env_idx = 1
        self.env_name = None
        self.use_emb = False

        # Training parameters
        self.num_episodes = 10000

        # Learning algorithm parameters
        self.replay_memory_size = 100000
        self.replay_memory_init_size = 50000
        self.update_target_estimator_every = 10000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 500000
        self.discount_factor = 0.99
        self.batch_size = 32
        self.record_video_every = 50

        # Model parameters
        self.embedding_dim = 300

        # Paths
        self.emb_path = None   # TODO: Change to real emb file
        self.vocab_path = "./data/vocab_10000.txt"
        self.experiment_dir = os.path.abspath("./experiments/{}".format(envs[self.env_idx]))
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "model")
        self.monitor_path = os.path.join(self.experiment_dir, "monitor")

        self.preprocess()

    def preprocess(self):
        self.env_name = envs[self.env_idx]
        if self.is_debug:
            self.replay_memory_init_size = 500
        if self.use_emb:
            self.emb_path = ""


def get_params():
    return Config()
