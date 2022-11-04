"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/2D-Quadcopter-AI

This is a slightly modified version of the DQN agent from LearnRL
I strongly recommend LearnRL as a reinforcement learning framework:
https://github.com/MathisFederico/LearnRL
"""

import learnrl as rl
import tensorflow as tf


class Memory:
    def __init__(self, max_memory_len):
        self.max_memory_len = max_memory_len
        self.memory_len = 0
        self.MEMORY_KEYS = (
            "observation",
            "action",
            "reward",
            "done",
            "next_observation",
        )
        self.datas = {key: None for key in self.MEMORY_KEYS}

    def remember(self, observation, action, reward, done, next_observation):
        for val, key in zip(
            (observation, action, reward, done, next_observation), self.MEMORY_KEYS
        ):
            batched_val = tf.expand_dims(val, axis=0)
            if self.memory_len == 0:
                self.datas[key] = batched_val
            else:
                self.datas[key] = tf.concat((self.datas[key], batched_val), axis=0)
            self.datas[key] = self.datas[key][-self.max_memory_len :]

        self.memory_len = len(self.datas[self.MEMORY_KEYS[0]])

    def sample(self, sample_size, method="random"):
        if method == "random":
            indexes = tf.random.shuffle(tf.range(self.memory_len))[:sample_size]
            datas = [tf.gather(self.datas[key], indexes) for key in self.MEMORY_KEYS]
        elif method == "last":
            datas = [self.datas[key][-sample_size:] for key in self.MEMORY_KEYS]
        else:
            raise ValueError(f"Unknowed method {method}")
        return datas

    def __len__(self):
        return self.memory_len


class Control:
    def __init__(self, exploration=0, exploration_decay=0, exploration_minimum=0):
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_minimum = exploration_minimum

    def update_exploration(self):
        self.exploration *= 1 - self.exploration_decay
        self.exploration = max(self.exploration, self.exploration_minimum)

    def act(self, Q):
        raise NotImplementedError(
            "You must define act(self, Q) when subclassing Control"
        )

    def __call__(self, Q, greedy):
        if greedy:
            return tf.argmax(Q, axis=-1, output_type=tf.int32)
        else:
            return self.act(Q)


class EpsGreedy(Control):
    def __init__(self, *args):
        super().__init__(*args)
        assert (
            self.exploration <= 1 and self.exploration >= 0
        ), "Exploration must be in [0, 1] for EpsGreedy"

    def act(self, Q):
        batch_size = Q.shape[0]
        action_size = Q.shape[1]

        actions_random = tf.random.uniform(
            (batch_size,), 0, action_size, dtype=tf.int32
        )
        actions_greedy = tf.argmax(Q, axis=-1, output_type=tf.int32)

        rd = tf.random.uniform((batch_size,), 0, 1)
        actions = tf.where(rd <= self.exploration, actions_random, actions_greedy)

        return actions


class Evaluation:
    def __init__(self, discount):
        self.discount = discount

    def eval(self, rewards, dones, next_observations, action_value):
        raise NotImplementedError("You must define eval when subclassing Evaluation")

    def __call__(self, rewards, dones, next_observations, action_value):
        return self.eval(rewards, dones, next_observations, action_value)


class QLearning(Evaluation):
    def eval(self, rewards, dones, next_observations, action_value):
        futur_rewards = rewards

        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            next_values = tf.reduce_max(
                action_value(next_observations[ndones]), axis=-1
            )

            ndones_indexes = tf.where(ndones)
            futur_rewards = tf.tensor_scatter_nd_add(
                futur_rewards, ndones_indexes, self.discount * next_values
            )

        return futur_rewards


class DQNAgent(rl.Agent):
    def __init__(
        self,
        action_value: tf.keras.Model = None,
        control: Control = None,
        memory: Memory = None,
        evaluation: Evaluation = None,
        sample_size=32,
        learning_rate=1e-4,
        training_period=4,
        update_period=1,
        update_factor=1,
        mem_method="random",
    ):

        self.action_value = action_value
        self.action_value_opt = tf.keras.optimizers.Adam(learning_rate)
        self.target_av = action_value
        self.learning_rate = learning_rate

        self.control = Control() if control is None else control
        self.memory = memory
        self.mem_method = mem_method
        self.evaluation = evaluation

        self.sample_size = sample_size

        self.step = 0
        self.training_period = training_period
        self.update_period = update_period
        self.update = update_factor

    @tf.function
    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)
        Q = self.action_value(observations)
        action = self.control(Q, greedy)[0]
        return action

    def learn(self):

        if self.step % self.update_period == 0:
            weights = self.action_value.get_weights()
            for i, target_weight in enumerate(self.target_av.get_weights()):
                weights[i] = (
                    self.update * weights[i] + (1 - self.update) * target_weight
                )
            self.target_av.set_weights(weights)

        if self.step % self.training_period != 0:
            return

        observations, actions, rewards, dones, next_observations = self.memory.sample(
            self.sample_size, method=self.mem_method
        )
        expected_futur_rewards = self.evaluation(
            rewards, dones, next_observations, self.target_av
        )

        with tf.GradientTape() as tape:
            Q = self.action_value(observations)

            action_index = tf.stack((tf.range(len(actions)), actions), axis=-1)
            Q_action = tf.gather_nd(Q, action_index)

            loss = tf.keras.losses.mse(expected_futur_rewards, Q_action)

        grads = tape.gradient(loss, self.action_value.trainable_weights)
        self.action_value_opt.apply_gradients(
            zip(grads, self.action_value.trainable_weights)
        )

        metrics = {
            "value": tf.reduce_mean(Q_action).numpy(),
            "loss": loss.numpy(),
            "exploration": self.control.exploration,
            "learning_rate": self.action_value_opt.lr.numpy(),
        }

        self.control.update_exploration()
        return metrics

    def remember(
        self, observation, action, reward, done, next_observation=None, info={}, **param
    ):
        self.memory.remember(observation, action, reward, done, next_observation)
        self.step += 1

    def save(self, filename):
        filename += ".h5"
        tf.keras.models.save_model(self.action_value, filename)
        print(f"Model saved at {filename}")

    def load(self, filename):
        self.action_value = tf.keras.models.load_model(
            filename, custom_objects={"tf": tf}
        )
        self.actor_opt = tf.optimizers.Adam(lr=self.learning_rate)
        self.action_value.compile(self.actor_opt)
        self.target_av = tf.keras.models.clone_model(self.action_value)


class ScoreCallback(rl.Callback):
    def __init__(self):
        self.score = 0

    def on_step_end(self, step, logs):
        self.score += self.playground.env.reward
