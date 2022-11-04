# **2D Quadcopter AI**

Controlling a 2D Quadcopter with rigidbody physics using Control Theory and Reinforcement Learning

The main game consists of controlling the drone to hit as many balloons within a time limit against AI drones.

<p align="center">
  <img src="media/main_game.gif" alt="Main Game" width="50%"/>
</p>

The currently implemented algorithms are:

- **Human**: Control of the propellers with the arrow keys
- **PID**: PID is a controller in control theory that uses the error between the drone position and the target position to calculate inputs
- **DQN**: Reinforcement learning agent that trained itself on multiple episodes of the game, by testing different actions and learning from the rewards it gets.

I added another game mode where the drone follows the mouse to move snow around a snowglobe.

<p align="center">
  <img src="media/snowglobe.gif" alt="Snowglobe" width="50%"/>
</p>
