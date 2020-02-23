import os

envs = {0: 'Breakout-v0',
        1: 'MontezumaRevenge-ram-v0',
        2: 'CartPole-v0'}

class Config:
    def __init__(self):

        # Modes
        self.is_train = False
        self.is_debug = True
        self.env_idx = 0
        self.env_name = None
        self.use_emb = False

        # Training parameters
        self.num_episodes = 10000

        # Learning algorithm parameters
        self.replay_memory_size = 500000
        self.replay_memory_init_size = 50000
        self.update_target_estimator_every = 10000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 500000
        self.discount_factor = 0.99
        self.batch_size = 32
        self.record_video_every = 50

        # Paths
        self.emb_path = "./data/emb_file"   # TODO: Change to real emb file
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
