"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/Quadcopter-AI

This is a gym environment based on drone_game (see Human/drone_game.py for details)
It is to be used with a DQN agent
The goal is to reach randomly positoned targets
"""

import os
from math import sin, cos, pi, sqrt
from random import randrange

import numpy as np
import gym
from gym import spaces

import pygame
from pygame.locals import *


class droneEnv(gym.Env):
    def __init__(self, render_every_frame, mouse_target):
        super(droneEnv, self).__init__()

        self.render_every_frame = render_every_frame
        # Makes the target follow the mouse
        self.mouse_target = mouse_target

        # Initialize Pygame, load sprites
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        self.FramePerSec = pygame.time.Clock()

        self.player = pygame.image.load(os.path.join("assets/sprites/drone_old.png"))
        self.player.convert()

        self.target = pygame.image.load(os.path.join("assets/sprites/target_old.png"))
        self.target.convert()

        pygame.font.init()
        self.myfont = pygame.font.SysFont("Comic Sans MS", 20)

        # Physics constants
        self.FPS = 60
        self.gravity = 0.08
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.003
        self.thruster_mean = 0.04
        self.mass = 1
        self.arm = 25

        # Initialize variables
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (400, 0, 0)
        (self.y, self.yd, self.ydd) = (400, 0, 0)
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        # Initialize game variables
        self.target_counter = 0
        self.reward = 0
        self.time = 0
        self.time_limit = 20
        if self.mouse_target is True:
            self.time_limit = 1000

        # 2 action thrust amplitude and thrust difference in float values between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        # 8 observations: angle_to_up, velocity, angle_velocity, distance_to_target, angle_to_target, angle_target_and_velocity, distance_to_target
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))

    def reset(self):
        # Reset variables
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (400, 0, 0)
        (self.y, self.yd, self.ydd) = (400, 0, 0)
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        self.target_counter = 0
        self.reward = 0
        self.time = 0

        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        """
        Calculates the observations

        Returns:
            np.ndarray: The normalized observations:
            - angle_to_up : angle between the drone and the up vector (to observe gravity)
            - velocity : velocity of the drone
            - angle_velocity : angle of the velocity vector
            - distance_to_target : distance to the target
            - angle_to_target : angle between the drone and the target
            - angle_target_and_velocity : angle between the to_target vector and the velocity vector
            - distance_to_target : distance to the target (HERE TWICE BY MISTAKE)
        """
        angle_to_up = self.a / 180 * pi
        velocity = sqrt(self.xd**2 + self.yd**2)
        angle_velocity = self.ad
        distance_to_target = (
            sqrt((self.xt - self.x) ** 2 + (self.yt - self.y) ** 2) / 500
        )
        angle_to_target = np.arctan2(self.yt - self.y, self.xt - self.x)
        # Angle between the to_target vector and the velocity vector
        angle_target_and_velocity = np.arctan2(
            self.yt - self.y, self.xt - self.x
        ) - np.arctan2(self.yd, self.xd)
        distance_to_target = (
            sqrt((self.xt - self.x) ** 2 + (self.yt - self.y) ** 2) / 500
        )
        return np.array(
            [
                angle_to_up,
                velocity,
                angle_velocity,
                distance_to_target,
                angle_to_target,
                angle_target_and_velocity,
                distance_to_target,
            ]
        ).astype(np.float32)

    def step(self, action):
        # Game loop
        self.reward = 0.0
        (action0, action1) = (action[0], action[1])

        # Act every 5 frames
        for _ in range(5):
            self.time += 1 / 60

            if self.mouse_target is True:
                self.xt, self.yt = pygame.mouse.get_pos()

            # Initialize accelerations
            self.xdd = 0
            self.ydd = self.gravity
            self.add = 0
            thruster_left = self.thruster_mean
            thruster_right = self.thruster_mean

            thruster_left += action0 * self.thruster_amplitude
            thruster_right += action0 * self.thruster_amplitude
            thruster_left += action1 * self.diff_amplitude
            thruster_right -= action1 * self.diff_amplitude

            # Calculating accelerations with Newton's laws of motions
            self.xdd += (
                -(thruster_left + thruster_right) * sin(self.a * pi / 180) / self.mass
            )
            self.ydd += (
                -(thruster_left + thruster_right) * cos(self.a * pi / 180) / self.mass
            )
            self.add += self.arm * (thruster_right - thruster_left) / self.mass

            self.xd += self.xdd
            self.yd += self.ydd
            self.ad += self.add
            self.x += self.xd
            self.y += self.yd
            self.a += self.ad

            dist = sqrt((self.x - self.xt) ** 2 + (self.y - self.yt) ** 2)

            # Reward per step survived
            self.reward += 1 / 60
            # Penalty according to the distance to target
            self.reward -= dist / (100 * 60)

            if dist < 50:
                # Reward if close to target
                self.xt = randrange(200, 600)
                self.yt = randrange(200, 600)
                self.reward += 100

            # If out of time
            if self.time > self.time_limit:
                done = True
                break

            # If too far from target (crash)
            elif dist > 1000:
                self.reward -= 1000
                done = True
                break

            else:
                done = False

            if self.render_every_frame is True:
                self.render("yes")

        info = {}

        return (
            self.get_obs(),
            self.reward,
            done,
            info,
        )

    def render(self, mode):
        # Pygame rendering
        pygame.event.get()
        self.screen.fill(0)
        self.screen.blit(
            self.target,
            (
                self.xt - int(self.target.get_width() / 2),
                self.yt - int(self.target.get_height() / 2),
            ),
        )
        player_copy = pygame.transform.rotate(self.player, self.a)
        self.screen.blit(
            player_copy,
            (
                self.x - int(player_copy.get_width() / 2),
                self.y - int(player_copy.get_height() / 2),
            ),
        )

        textsurface = self.myfont.render(
            "Collected: " + str(self.target_counter), False, (255, 255, 255)
        )
        self.screen.blit(textsurface, (20, 20))
        textsurface3 = self.myfont.render(
            "Time: " + str(int(self.time)), False, (255, 255, 255)
        )
        self.screen.blit(textsurface3, (20, 50))

        pygame.display.update()
        self.FramePerSec.tick(self.FPS)

    def close(self):
        pass
