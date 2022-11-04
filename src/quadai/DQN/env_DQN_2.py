"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/2D-Quadcopter-AI

This is a gym environment based on drone_game (see drone_game.py for details)
It is to be used with a DQN agent
The goal is to collect as many targets as you can within the time limit
"""

import numpy as np
import gym
from gym import spaces

import pygame
import os
from pygame.locals import *
from math import sin, cos, pi, sqrt
from random import randrange


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
        self.diff_amplitude = 0.0005
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
        self.hover_timer_default = 240
        self.hover_timer = self.hover_timer_default
        self.on_target = False
        self.target_counter = 0
        self.reward = 0
        self.time = 0
        self.time_limit = 20
        if self.mouse_target == True:
            self.time_limit = 1000

        # 5 actions: Nothing, Up, Down, Right, Left
        self.action_space = gym.spaces.Discrete(5)
        # 6 observations: x_to_target, x_speed, y_to_target, y_speed, angle, angle_speed
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

    def reset(self):
        # Reset variables
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (400, 0, 0)
        (self.y, self.yd, self.ydd) = (400, 0, 0)
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        self.hover_timer_default = 240
        self.hover_timer = self.hover_timer_default
        self.on_target = False
        self.target_counter = 0
        self.reward = 0
        self.time = 0
        return np.array(
            [self.xt - self.x, self.xd, self.yt - self.y, self.yd, self.a, self.ad]
        ).astype(np.float32)

    def step(self, action):
        # Game loop
        self.reward = 0.0
        action = int(action)

        # Act every 5 frames
        for i in range(5):
            self.time += 1 / 60

            if self.mouse_target == True:
                self.xt, self.yt = pygame.mouse.get_pos()

            # Initialize accelerations
            self.xdd = 0
            self.ydd = self.gravity
            self.add = 0
            thruster_left = self.thruster_mean
            thruster_right = self.thruster_mean

            if action == 0:
                pass
            elif action == 1:
                thruster_left += self.thruster_amplitude
                thruster_right += self.thruster_amplitude
            elif action == 2:
                thruster_left -= self.thruster_amplitude
                thruster_right -= self.thruster_amplitude
            elif action == 3:
                thruster_left += self.diff_amplitude
                thruster_right -= self.diff_amplitude
            elif action == 4:
                thruster_left -= self.diff_amplitude
                thruster_right += self.diff_amplitude

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
            self.reward -= dist * 0.5 / (1000 * 60)

            if dist < 50:
                self.on_target = True
                # Reward if close to target
                self.reward += 10 / 60

            else:
                self.on_target = False
                self.hover_timer = self.hover_timer_default

            if self.on_target == True:
                self.hover_timer -= 1

            if self.hover_timer < 0:
                self.reward += 50
                self.hover_timer = self.hover_timer_default
                self.xt = randrange(200, 600)
                self.yt = randrange(200, 600)

            # If out of time
            if self.time > self.time_limit:
                done = True
                # Reward for surviving
                self.reward += 10
                break

            # If too far from target (crash)
            elif dist > 1000:
                done = True
                break

            else:
                done = False

            if self.render_every_frame == True:
                self.render("yes")

        info = {}

        return (
            np.array(
                [self.xt - self.x, self.xd, self.yt - self.y, self.yd, self.a, self.ad]
            ).astype(np.float32),
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
