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

# Compare runs according to learning rate
for learning_rate in np.logspace(-4, -1, 10):
    # Keep two decimals
    round_lr = round(learning_rate, 5)
    run = wandb.init(
        project="quadai-lr",
        sync_tensorboard=True,
        monitor_gym=True,
        name=f"lr_{round_lr}",
        config={
            "learning_rate": learning_rate,
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
        learning_rate=learning_rate,
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, save_path=log_dir, name_prefix="rl_model_lr"
    )

    # Train the agent
    model.learn(
        total_timesteps=500000,
        callback=[
            checkpoint_callback,
            WandbCallback(
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




