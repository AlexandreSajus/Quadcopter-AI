"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/Quadcopter-AI

This is a helpful way to test env_DQN
"""

from env_DQN import droneEnv

env = droneEnv(True, False)

obs = env.reset()
env.render("yes")

print("Observation space:")
print(env.observation_space)
print("")
print("Action space:")
print(env.action_space)
print("")
print("Action space sample:")
print(env.action_space.sample())

# Choose an action to execute n_steps times
action = 4
n_steps = 20
for step in range(n_steps):
    print("Step {}".format(step + 1))
    obs, reward, done, info = env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    env.render("yes")
    if done:
        print("Done!", "reward=", reward)
        break
