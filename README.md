# **2D Quadcopter AI**

Controlling a 2D Quadcopter with rigidbody physics using Control Theory and Reinforcement Learning

The currently implemented algorithms are:

- **Human**: discrete control of the propellers with the arrow keys
- **DQN**: DQN is a reinforcement learning agent that trained itself on multiple episodes of the game, by testing different actions and learning from the rewards it gets.
- **PID**: PID is a controller in control theory that uses the error between the drone position and the target position to calculate the force to apply to the drone.

The main game consists of controlling the drone to hit as many balloons within a time limit. Getting out of bounds puts the drone on a respawn timer

<p align="center">
  <img src="media/main_game.gif" alt="Main Game" width="50%"/>
</p>

I added another game mode where the human just controls the drone with the arrow keys to move snow particles in a snowglobe.

<p align="center">
  <img src="media/snowglobe.gif" alt="Snowglobe" width="50%"/>
</p>

## Installation

- Install requirements.txt
- Run main.py or snowglobe.py
