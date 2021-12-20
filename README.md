# 2D-Quadcopter-AI

Controlling a 2D Quadcopter with rigidbody physics using Control Theory and Reinforcement Learning

The full scope of this project is to benchmark different ways to control a quadcopter in a 2D simulated environment:

- **Human**: discrete control of the propellers with the arrow keys
- **DQN**: fully emergent behavior, no imitation learning, same controls as human
- **PID**: cascade control with multiple PIDs
- **GenPID**: PID Tuning using genetic algorithms (not yet implemented)
- **RLPID**: using Reinforcement Learning (DDPG) to set the PID target in real time (not yet implemented)

Here is the main game where an expert highly trained human player (me...) gets destroyed by a PID agent (and there's also a DUMB DQN AGENT THAT NEVER LEARNS)

![](media/main_game.gif)

Oh also I got bored so I made a christmas-themed mod in snowglobe.py:

![](media/snowglobe.gif)

## Setup

- Install requirements.txt
- Run main.py
