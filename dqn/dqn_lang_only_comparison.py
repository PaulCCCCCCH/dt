import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import plotting
import params
from collections import deque, namedtuple
import time

if "../" not in sys.path:
    sys.path.append("../")

_config = params.get_params()

_env = gym.envs.make(_config.env_name)

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
# VALID_ACTIONS = [0, 1, 2, 3]
VALID_ACTIONS = list(range(_env.action_space.n))

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(_env.spec.id))
checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "model")
monitor_path = os.path.join(experiment_dir, "monitor")

_config.experiment_dir = experiment_dir
_config.checkpoint_dir = checkpoint_dir
_config.checkpoint_path = checkpoint_path
_config.monitor_path = monitor_path

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(monitor_path):
    os.makedirs(monitor_path)

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

IS_TRAIN = _config.is_train

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self, env):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


class Estimator():
    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)

        # TODO: Language generation model here:
        with tf.variable_scope("language_model"):
            pass

        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


class Runner:
    def __init__(self,
                 sess,
                 env,
                 saver,
                 q_estimator,
                 target_estimator,
                 state_processor,
                 config):
        self.sess = sess
        self.env = env
        self.saver = saver
        self.q_estimator = q_estimator
        self.target_estimator = target_estimator
        self.params = config
        self.replay_memory = []
        self.state_processor = state_processor
        self.stats = plotting.EpisodeStats(
                episode_lengths=np.zeros(config.num_episodes),
                episode_rewards=np.zeros(config.num_episodes))

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)

        self.total_t = sess.run(tf.contrib.framework.get_global_step())

        self.epsilons = np.linspace(config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)
        self.policy = make_epsilon_greedy_policy(
            q_estimator,
            len(VALID_ACTIONS))

    def populate_memory(self):

        print("Populating replay memory...")
        state = self.env.reset()
        state = self.state_processor.process(self.sess, state)
        state = np.stack([state] * 4, axis=2)
        for i in range(self.params.replay_memory_init_size):
            action_probs = self.policy(self.sess, state, self.epsilons[min(self.total_t, self.params.epsilon_decay_steps-1)])
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = self.env.step(VALID_ACTIONS[action])
            next_state = self.state_processor.process(self.sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            self.replay_memory.append(Transition(state, action, reward, next_state, done))
            if done:
                state = self.env.reset()
                state = self.state_processor.process(self.sess, state)
                state = np.stack([state] * 4, axis=2)
            else:
                state = next_state


    def train(self):

        self.env = Monitor(self.env,
                           directory=monitor_path,
                           resume=True,
                           video_callable=lambda count: count % self.params.record_video_every == 0)

        for i_episode in range(self.params.num_episodes):

            # Save the current checkpoint
            self.saver.save(tf.get_default_session(), checkpoint_path)

            # Reset the environment
            state = self.env.reset()
            state = self.state_processor.process(self.sess, state)
            state = np.stack([state] * 4, axis=2)
            loss = None

            # One step in the environment
            for t in itertools.count():

                # Epsilon for this time step
                epsilon = self.epsilons[min(self.total_t, self.params.epsilon_decay_steps-1)]

                # Add epsilon to Tensorboard
                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=epsilon, tag="epsilon")
                self.q_estimator.summary_writer.add_summary(episode_summary, self.total_t)

                # Maybe update the target estimator
                if self.total_t % self.params.update_target_estimator_every == 0:
                    copy_model_parameters(self.sess, self.q_estimator, self.target_estimator)
                    print("\nCopied model parameters to target network.")

                # Print out which step we're on, useful for debugging.
                print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                        t, self.total_t, i_episode + 1, self.params.num_episodes, loss), end="")

                sys.stdout.flush()

                # Take a step
                action_probs = self.policy(self.sess, state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(VALID_ACTIONS[action])
                next_state = self.state_processor.process(self.sess, next_state)
                next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

                # If our replay memory is full, pop the first element
                if len(self.replay_memory) == self.params.replay_memory_size:
                    self.replay_memory.pop(0)

                # Save transition to replay memory
                self.replay_memory.append(Transition(state, action, reward, next_state, done))

                # Update statistics
                self.stats.episode_rewards[i_episode] += reward
                self.stats.episode_lengths[i_episode] = t

                # Sample a minibatch from the replay memory
                samples = random.sample(self.replay_memory, self.params.batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                # Calculate q values and targets (Double DQN)
                q_values_next = self.q_estimator.predict(self.sess, next_states_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = self.target_estimator.predict(self.sess, next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                    self.params.discount_factor * q_values_next_target[np.arange(self.params.batch_size), best_actions]

                # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = self.q_estimator.update(self.sess, states_batch, action_batch, targets_batch)

                if done:
                    break

                state = next_state
                self.total_t += 1

            # Add summaries to tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=self.stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
            episode_summary.value.add(simple_value=self.stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
            self.q_estimator.summary_writer.add_summary(episode_summary, self.total_t)
            self.q_estimator.summary_writer.flush()

            yield self.total_t, plotting.EpisodeStats(
                episode_lengths=self.stats.episode_lengths[:i_episode+1],
                episode_rewards=self.stats.episode_rewards[:i_episode+1])

        self.env.monitor.close()
        return self.stats

    def run(self):
        # print(env.action_space)
        # print(env.observation_space)
        for i_episode in range(1):
            state = self.env.reset()
            state = self.state_processor.process(self.sess, state)
            state = np.stack([state] * 4, axis=2)
            for t in range(10000):
                self.env.render()
                # Get action
                action_probs = self.policy(self.sess, state,
                                           self.epsilons [min(self.total_t, self.params.epsilon_decay_steps - 1)])
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(VALID_ACTIONS[action])
                next_state = self.state_processor.process(self.sess, next_state)
                next_state = np.append(state [:, :, 1:], np.expand_dims(next_state, 2), axis=2)
                self.replay_memory.append(Transition(state, action, reward, next_state, done))
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                else:
                    state = next_state
                time.sleep(1/24)
                if done:
                    break

tf.reset_default_graph()

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
_q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
_target_estimator = Estimator(scope="target_q")

# State processor
_state_processor = StateProcessor(_env)

with tf.Session() as _sess:
    _sess.run(tf.global_variables_initializer())
    # Create directories for checkpoints and summaries

    _saver = tf.train.Saver()
    # Load a previous checkpoint if we find one

    runner = Runner(_sess,
                    _env,
                    _saver,
                    q_estimator=_q_estimator,
                    target_estimator=_target_estimator,
                    state_processor=_state_processor,
                    config=_config)

    if IS_TRAIN:
        runner.populate_memory()
        for t, stats in runner.train():
            print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))

    else:
        runner.run()

_env.close()
