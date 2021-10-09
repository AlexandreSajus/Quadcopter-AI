import numpy as np
import gym
from gym import spaces

import pygame
import sys
import os
from pygame.locals import *
from math import sin, cos, pi, sqrt
from random import randrange


class droneEnv(gym.Env):

    def __init__(self):
        super(droneEnv, self).__init__()

        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        self.FramePerSec = pygame.time.Clock()

        self.player = pygame.image.load(os.path.join("drone.png"))
        self.player.convert()

        self.target = pygame.image.load(os.path.join("target.png"))
        self.target.convert()

        self.FPS = 60
        self.gravity = 0.08
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.005
        self.thruster_mean = 0.04
        self.mass = 1
        self.arm = 25

        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (400, 0, 0)
        (self.y, self.yd, self.ydd) = (400, 0, 0)
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 20)

        self.target_counter = 0
        self.reward = 0
        self.time = 0
        self.time_limit = 20

        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(8,))

    def reset(self):
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (400, 0, 0)
        (self.y, self.yd, self.ydd) = (400, 0, 0)
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        self.target_counter = 0
        self.reward = 0
        self.time = 0
        return np.array([self.x, self.xd, self.y, self.yd, self.a, self.ad, self.xt - self.x, self.yt - self.y]).astype(np.float32)

    def step(self, action):
        self.time += 1/60

        self.reward = 0.0

        self.xdd = 0
        self.ydd = self.gravity
        self.add = 0
        thruster_left = self.thruster_mean
        thruster_right = self.thruster_mean

        action = np.array(action)

        thruster_left += self.thruster_amplitude*action[0]
        thruster_right += self.thruster_amplitude*action[0]
        thruster_left += self.diff_amplitude*action[1]
        thruster_right -= self.diff_amplitude*action[1]

        self.xdd += -(thruster_left + thruster_right) * \
            sin(self.a*pi/180)/self.mass
        self.ydd += -(thruster_left + thruster_right) * \
            cos(self.a*pi/180)/self.mass
        self.add += self.arm*(thruster_right - thruster_left)/self.mass

        self.xd += self.xdd
        self.yd += self.ydd
        self.ad += self.add
        self.x += self.xd
        self.y += self.yd
        self.a += self.ad

        dist = sqrt((self.x - self.xt)**2 + (self.y - self.yt)**2)
        if dist < 50:
            self.xt = randrange(200, 600)
            self.yt = randrange(200, 600)
            self.target_counter += 1
            self.reward += 10

        if self.x < 0 or self.y < 0 or self.x > 800 or self.y > 800 or self.time > self.time_limit:
            self.reward -= 10*(self.time_limit - self.time)
            done = True
        else:
            done = False

        info = {}

        return np.array([self.x, self.xd, self.y, self.yd, self.a, self.ad, self.xt - self.x, self.yt - self.y]).astype(np.float32), self.reward, done, info

    def render(self, mode):
        pygame.event.get()
        self.screen.fill(0)
        self.screen.blit(self.target, (self.xt - int(self.target.get_width()/2),
                                       self.yt - int(self.target.get_height()/2)))
        player_copy = pygame.transform.rotate(self.player, self.a)
        self.screen.blit(player_copy, (self.x - int(player_copy.get_width()/2),
                                       self.y - int(player_copy.get_height()/2)))

        textsurface = self.myfont.render(
            'Collected: ' + str(self.target_counter), False, (255, 255, 255))
        self.screen.blit(textsurface, (20, 20))
        textsurface3 = self.myfont.render(
            'Time: ' + str(int(self.time)), False, (255, 255, 255))
        self.screen.blit(textsurface3, (20, 50))

        pygame.display.update()
        self.FramePerSec.tick(self.FPS)

    def close(self):
        pass
