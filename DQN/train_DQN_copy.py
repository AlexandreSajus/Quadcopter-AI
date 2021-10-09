"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/2D-Quadcopter-AI

This is a copy of train_DQN
It is used to measure training efficiency over time
"""

import numpy as np
import learnrl as rl
import tensorflow as tf
from DQN.env_DQN import droneEnv
from DQN.agent_DQN import DQNAgent, Memory, EpsGreedy, QLearning, ScoreCallback

kl = tf.keras.layers

n_episodes = 10

scores = []

gens = [40,27,24,23]
i = -1

for model in [4,7,8,9]:
    i += 1
    limit = gens[i]
    for gen in range(limit):

        class Config():
            def __init__(self, config):
                for key, val in config.items():
                    setattr(self, key, val)


        config = {
            'model_name': 'newborn',
            'max_memory_len': 40960,

            'exploration': 1.0,
            'exploration_decay': 6e-7,
            'exploration_minimum': 0.2,

            'discount': 0.85,

            'dense_1_size': 256,
            'dense_1_activation': 'relu',
            'dense_2_size': 128,
            'dense_2_activation': 'relu',
            'dense_3_size': 128,
            'dense_3_activation': 'relu',
            'dense_4_size': 64,
            'dense_4_activation': 'relu',

            'sample_size': 4096,
            'learning_rate': 2.2e-5,

            'training_period': 1,
            'update_period': 20,
            'update_factor': 0.2,

            'mem_method': 'random',

            'render_every_frame': False,
            'mouse_target': False,
            'test_only': True,

            'generations': 1000,
            'episodes_per_gen': 1000,
            'test_episodes': 5,

            'load_model': True,
            'load_path': "models/DQN/models8/newborn01gen23.h5",
            'save_path': "models/DQN/models9/newborn01gen"
        }

        config = Config(config)

        env = droneEnv(render_every_frame=config.render_every_frame,
                    mouse_target=config.mouse_target)

        memory = Memory(config.max_memory_len)
        control = EpsGreedy(
            config.exploration,
            config.exploration_decay,
            config.exploration_minimum
        )
        evaluation = QLearning(config.discount)

        init_re = tf.keras.initializers.HeUniform()
        init_th = tf.keras.initializers.GlorotUniform()

        inputs = tf.keras.Input(shape=(6,))
        x = kl.Dense(config.dense_1_size, activation=config.dense_1_activation,
                    kernel_initializer=init_re)(inputs)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.3)(x, training=False)
        x = kl.Dense(config.dense_2_size, activation=config.dense_2_activation,
                    kernel_initializer=init_re)(x)
        x = kl.Dense(config.dense_3_size, activation=config.dense_3_activation,
                    kernel_initializer=init_re)(x)
        x = kl.Dense(config.dense_4_size, activation=config.dense_4_activation,
                    kernel_initializer=init_re)(x)
        outputs = kl.Dense(5, activation='linear',
                        kernel_initializer=init_re)(x)
        action_value = tf.keras.Model(inputs=inputs, outputs=outputs)

        agent = DQNAgent(
            action_value=action_value,
            control=control,
            memory=memory,
            evaluation=evaluation,
            sample_size=config.sample_size,
            learning_rate=config.learning_rate,
            training_period=config.training_period,
            update_period=config.update_period,
            update_factor=config.update_factor,
            mem_method=config.mem_method
        )

        metrics = [
            ('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
            ('handled_reward~reward', {'steps': 'sum', 'episode': 'sum'}),
            'loss',
            'exploration~exp',
            'value~Q'
        ]

        pg = rl.Playground(env, agent)

        path = "models/DQN/models" + str(model) + "/newborn01gen" + str(gen) + ".h5"
        
        agent.load(path)

        score = ScoreCallback()

        pg.test(n_episodes, verbose=0, episodes_cycle_len=1,
                callbacks=[score])

        scores.append(int(score.score*10**2/n_episodes)/10**2)

        print("Score gen " + str(gen) + " : " + str(scores[-1]))

np.save("temp", np.asarray(scores))
print(scores)
