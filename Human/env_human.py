"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/2D-Quadcopter-AI

This is a simple game where you control a 2D Quadcopter with arrow keys
The goal is to reach the target and hover around it
"""

import pygame
import os
from pygame.locals import *
from math import sin, cos, pi, sqrt
from random import randrange

# Game constants
FPS = 60
WIDTH = 800
HEIGHT = 800

# Physics constants
gravity = 0.08
# Propeller force for UP and DOWN
thruster_amplitude = 0.04
# Propeller force for LEFT and RIGHT rotations
diff_amplitude = 0.0015
# By default, thruster will apply a force of thruster_mean
thruster_mean = 0.04
mass = 1
# Length from center of mass to propeller
arm = 25

# Initialize Pygame, load sprites
FramePerSec = pygame.time.Clock()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

player = pygame.image.load(os.path.join("assets/sprites/drone_old.png"))
player.convert()

target = pygame.image.load(os.path.join("assets/sprites/target_old.png"))
target.convert()

pygame.font.init()
myfont = pygame.font.SysFont("Comic Sans MS", 20)

# Initialize physics variables (a=angle) (xt,yt=target_coords)
(a, ad, add) = (0, 0, 0)
(x, xd, xdd) = (400, 0, 0)
(y, yd, ydd) = (400, 0, 0)
xt = randrange(200, 600)
yt = randrange(200, 600)

# Initialize game variables
target_counter = 0
reward = 0
time = 0
time_limit = 20

# Game loop
while True:
    pygame.event.get()
    screen.fill(0)

    time += 1 / 60

    # Initialize accelerations
    xdd = 0
    ydd = gravity
    add = 0

    # Calculate propeller force in function of input
    thruster_left = thruster_mean
    thruster_right = thruster_mean
    pressed_keys = pygame.key.get_pressed()
    if pressed_keys[K_UP]:
        thruster_left += thruster_amplitude
        thruster_right += thruster_amplitude
    if pressed_keys[K_DOWN]:
        thruster_left -= thruster_amplitude
        thruster_right -= thruster_amplitude
    if pressed_keys[K_LEFT]:
        thruster_left -= diff_amplitude
    if pressed_keys[K_RIGHT]:
        thruster_right -= diff_amplitude

    # Calculate accelerations according to Newton's laws of motion
    xdd += -(thruster_left + thruster_right) * sin(a * pi / 180) / mass
    ydd += -(thruster_left + thruster_right) * cos(a * pi / 180) / mass
    add += arm * (thruster_right - thruster_left) / mass

    # Calculate speed
    xd += xdd
    yd += ydd
    ad += add

    # Calculate position
    x += xd
    y += yd
    a += ad

    # Calculate distance to target
    dist = sqrt((x - xt) ** 2 + (y - yt) ** 2)
    reward += 1 / 60
    reward -= dist * 0.5 / (1000 * 60)
    # If target reached, respawn target
    if dist < 50:
        reward += 10 / 60

    # If out of time
    if time > time_limit:
        done = True
        # Reward for surviving
        reward += 10
        pygame.quit()
        break

    # If too far from target (crash)
    elif dist > 1000:
        pygame.quit()
        break

    screen.blit(
        target, (xt - int(target.get_width() / 2), yt - int(target.get_height() / 2))
    )
    player_copy = pygame.transform.rotate(player, a)
    screen.blit(
        player_copy,
        (x - int(player_copy.get_width() / 2), y - int(player_copy.get_height() / 2)),
    )

    # Update Pygame screen
    textsurface2 = myfont.render(
        "Reward: " + str(int(reward * 10) / 10), False, (255, 255, 255)
    )
    screen.blit(textsurface2, (20, 50))
    textsurface3 = myfont.render("Time: " + str(int(time)), False, (255, 255, 255))
    screen.blit(textsurface3, (20, 80))

    pygame.display.update()
    FramePerSec.tick(FPS)

print("Reward: " + str(reward))
