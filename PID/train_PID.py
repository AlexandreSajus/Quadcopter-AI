"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/2D-Quadcopter-AI

This is where you can train and test PID agents
Just modify the PID values and run
"""

from PID.env_PID_3 import droneEnv
from PID.controller_PID import PID

env = droneEnv(render_every_frame=True, mouse_target=False)

xPID = PID(0.2,0,0.2,25,-25)
aPID = PID(0.02,0,0.01,1,-1)

yPID = PID(2.5,0,1.5,100,-100)
ydPID = PID(1,0,0,1,-1)

obs = env.reset()
env.render("yes")

dt = 1/60
n_steps = 100000
for step in range(n_steps):
    [error_x, xd, error_y, yd, a, ad] = obs

    ac = xPID.compute(-error_x, dt)

    error_a = ac - a
    action1 = aPID.compute(-error_a, dt)

    ydc = yPID.compute(error_y, dt)
    error_yd = ydc - yd
    action0 = ydPID.compute(-error_yd, dt)

    """if step % 60 == 1:
        print(yd)
        print(action0)
        print("")"""

    obs, reward, done, info = env.step([action0, action1])
    env.render("yes")
    if done:
        break