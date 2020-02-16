envs = {0: 'Breakout-v0',
        1: 'MontezumaRevenge-ram-v0',
        2: 'CartPole-v0'}

class Config:
    def __init__(self):
        self.num_episodes = 10000
        self.replay_memory_size = 500000
        self.replay_memory_init_size = 50000
        self.update_target_estimator_every = 10000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 500000
        self.discount_factor = 0.99
        self.batch_size = 32
        self.record_video_every = 50
        self.is_train = True
        self.is_debug = True
        self.env_idx = 0

        self.preprocess()

    def preprocess(self):
        self.env_name = envs[self.env_idx]
        if self.is_debug:
            self.replay_memory_init_size = 500


def get_params():
    return Config()
