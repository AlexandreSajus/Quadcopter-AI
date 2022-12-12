"""
Test a trained SAC agent on the droneEnv environment
"""

import os
import gym
import numpy as np
import torch as th
from stable_baselines3 import SAC

from env_SAC import droneEnv

MODEL_PATH = "models/sac_model_v2_5000000_steps.zip"

# Create and wrap the environment
env = droneEnv(True, False)

# Load the trained agent
model = SAC.load(MODEL_PATH, env=env)

# Evaluate the agent
for i in range(10):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    print("Episode reward", episode_reward)
    env.render("yes")
