# **2D Quadcopter AI**

Controlling a 2D Quadcopter with Rigidbody Physics using Control Theory and Reinforcement Learning

<p align="center">
  <img src="media/main_game.gif" alt="Main Game" width="50%"/>
</p>

The main game consists of controlling the drone to hit as many balloons within a time limit against AI drones.

The currently implemented algorithms are:

- **Human**: Control of the propellers with the arrow keys
- **PID**: Controller in control theory that uses the error between the drone position and the target position to calculate inputs
- **DQN**: Reinforcement Learning agent that trained itself on multiple episodes of the game, by testing different actions and learning from the rewards it gets.

I added another game mode where the drone follows the mouse to move snow around a snowglobe.

<p align="center">
  <img src="media/snowglobe.gif" alt="Snowglobe" width="50%"/>
</p>

## **Installation** as a Windows Executable

I have published the project as a game on itch.io here: https://alexandresajus.itch.io/2d-quadcopter

## **Installation** in Python

Make sure you have Python installed on your computer. Then, in a terminal, run the following commands:

### **1. Install the package with pip in your terminal:**

```bash
pip install git+https://github.com/AlexandreSajus/2D-Quadcopter-AI.git
```

### **2. Run the game:**

**If you want to run the balloon game:**

```bash
python -m quadai balloon
```

- Control your drone using the arrow keys
- The drone is very sensitive so tap the keys slowly
- Reach as many balloons as you can within the time limit

**If you want to run the snowglobe game:**

```bash
python -m quadai snowglobe
```

- Control the drone using your mouse
- The drone's airflow will move the snow around