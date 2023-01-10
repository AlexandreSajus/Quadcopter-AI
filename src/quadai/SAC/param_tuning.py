"""
Train an SAC agent using sb3 on the droneEnv environment
"""

import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from env_SAC import droneEnv

params = ["gamma", "learning_rate", "buffer_size", "tau", "batch_size"]
gamma_range = []
learning_rate_range = []
buffer_size_range = [1, 500, 5000, 50000, 500000]
tau_range = [0.00001, 0.001, 0.1, 0.5, 0.99]
batch_size_range = [1, 32, 64, 128, 256]
ranges = [
    gamma_range,
    learning_rate_range,
    buffer_size_range,
    tau_range,
    batch_size_range,
]
defaults = [0.99, 0.0003, 50000, 0.005, 64]

for i in range(len(params)):
    for j in range(len(ranges[i])):
        # Set hyperparameters
        gamma = defaults[0]
        learning_rate = defaults[1]
        buffer_size = defaults[2]
        tau = defaults[3]
        batch_size = defaults[4]

        if params[i] == "gamma":
            gamma = ranges[i][j]
        elif params[i] == "learning_rate":
            learning_rate = ranges[i][j]
        elif params[i] == "buffer_size":
            buffer_size = ranges[i][j]
        elif params[i] == "tau":
            tau = ranges[i][j]
        elif params[i] == "batch_size":
            batch_size = ranges[i][j]

        run = wandb.init(
            # CHANGE THIS to quadai-params
            project="quadai-params",
            sync_tensorboard=True,
            monitor_gym=True,
            name=f"{params[i]}_{ranges[i][j]}",
            config={
                "gamma": gamma,
                "learning_rate": learning_rate,
                "buffer_size": buffer_size,
                "tau": tau,
                "batch_size": batch_size,
            },
        )

        # Create log dir
        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)

        # Create and wrap the environment
        env = droneEnv(False, False)
        env = Monitor(env, log_dir)

        # Create SAC agent
        model = SAC(
            "MlpPolicy",
            env,
            verbose=2,
            tensorboard_log=log_dir,
            gamma=gamma,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            tau=tau,
            batch_size=batch_size,
        )

        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=100000, save_path=log_dir, name_prefix="rl_model_lr"
        )

        # Train the agent
        model.learn(
            # CHANGE THIS TO 500000
            total_timesteps=500000,
            callback=[
                checkpoint_callback,
                WandbCallback(
                    # CHANGE THIS TO 5000
                    gradient_save_freq=5000,
                    model_save_path=f"models/{run.id}",
                    model_save_freq=100000,
                    verbose=2,
                ),
            ],
        )

        # Close
        env.close()
        run.finish()
