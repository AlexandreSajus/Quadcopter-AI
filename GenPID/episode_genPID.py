"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/2D-Quadcopter-AI

This is where you can train and test PID agents
Just modify the PID values and run
"""

from PID.controller_PID import PID


def episode(env, n_episodes, xPID_params, aPID_params, yPID_params, ydPID_params):
    xPID = PID(xPID_params)
    aPID = PID(aPID_params)

    yPID = PID(yPID_params)
    ydPID = PID(ydPID_params)

    for i in range(n_episodes):

        obs = env.reset()
        env.render("yes")

        total_reward = 0

        dt = 1 / 60
        n_steps = 200 * 60
        for step in range(n_steps):
            [error_x, xd, error_y, yd, a, ad] = obs

            ac = xPID.compute(-error_x, dt)

            error_a = ac - a
            action1 = aPID.compute(-error_a, dt)

            ydc = yPID.compute(error_y, dt)
            error_yd = ydc - yd
            action0 = ydPID.compute(-error_yd, dt)

            obs, reward, done, info = env.step([action0, action1])
            total_reward += reward

            env.render("yes")
            if done:
                break

    return total_reward / 10
