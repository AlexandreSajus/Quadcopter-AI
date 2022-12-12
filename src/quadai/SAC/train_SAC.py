"""
Train an SAC agent using sb3 on the droneEnv environment
"""

import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from env_SAC import droneEnv

run = wandb.init(
    project="quadai",
    sync_tensorboard=True,
    monitor_gym=True,
)

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = droneEnv(False, False)
env = Monitor(env, log_dir)

# Create SAC agent
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=100000, save_path=log_dir, name_prefix="rl_model_v2"
)

# Train the agent
model.learn(
    total_timesteps=10000000,
    callback=[
        checkpoint_callback,
        WandbCallback(
            gradient_save_freq=100000,
            model_save_path=f"models/{run.id}",
            model_save_freq=100000,
            verbose=2,
        ),
    ],
)
