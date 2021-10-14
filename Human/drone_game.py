"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/2D-Quadcopter-AI

This is a simple game where you control a 2D Quadcopter with arrow keys
The goal is to reach as many targets as possible within the time limit
This is used as a base to create a simulation environment
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
diff_amplitude = 0.003
# By default, thruster will apply a force of thruster_mean
thruster_mean = 0.04
mass = 1
# Length from center of mass to propeller
arm = 25

# Initialize Pygame, load sprites
FramePerSec = pygame.time.Clock()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

player_width = 80
player_speed = 0.3
player = []
for i in range(1,5):
    image = pygame.image.load(os.path.join("assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-" + str(i) + ".png"))
    image.convert()
    player.append(pygame.transform.scale(image,(player_width,int(player_width*0.30))))

target_width = 30
target_speed = 0.1
target = []
for i in range(1,8):
    image = pygame.image.load(os.path.join("assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/red-plain-" + str(i) + ".png"))
    image.convert()
    target.append(pygame.transform.scale(image,(target_width,int(target_width*1.73))))

pygame.font.init()
myfont = pygame.font.Font('assets/fonts/Roboto-Regular.ttf', 30)

# Initialize physics variables (a=angle) (xt,yt=target_coords)
(a, ad, add) = (0, 0, 0)
(x, xd, xdd) = (400, 0, 0)
(y, yd, ydd) = (400, 0, 0)
xt = randrange(200, 600)
yt = randrange(200, 600)

# Initialize game variables
target_counter = 0
time = 0
step = 0
time_limit = 30

# Game loop
while True:
    pygame.event.get()
    screen.fill((131,176,181))

    time += 1/60
    step += 1

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
    xdd += -(thruster_left + thruster_right)*sin(a*pi/180)/mass
    ydd += -(thruster_left + thruster_right)*cos(a*pi/180)/mass
    add += arm*(thruster_right - thruster_left)/mass

    # Calculate speed
    xd += xdd
    yd += ydd
    ad += add

    # Calculate position
    x += xd
    y += yd
    a += ad

    # Calculate distance to target
    dist = sqrt((x - xt)**2 + (y - yt)**2)
    # If target reached, respawn target
    if dist < 50:
        xt = randrange(200, 600)
        yt = randrange(200, 600)
        target_counter += 1

    # Failure conditions
    if dist > 1000 or time > time_limit:
        break

    target_sprite = target[int(step*target_speed)%len(target)]
    screen.blit(target_sprite, (xt - int(target_sprite.get_width()/2),
                yt - int(target_sprite.get_height()/2)))
    player_sprite = player[int(step*player_speed)%len(player)]
    player_copy = pygame.transform.rotate(player_sprite, a)
    screen.blit(player_copy, (x - int(player_copy.get_width()/2),
                y - int(player_copy.get_height()/2)))

    # Update Pygame screen
    textsurface = myfont.render(
        'Collected : ' + str(target_counter), True, (255, 255, 255))
    screen.blit(textsurface, (20, 20))
    textsurface3 = myfont.render(
        'Time : ' + str(int(time_limit - time)), True, (255, 255, 255))
    screen.blit(textsurface3, (20, 60))

    pygame.display.update()
    FramePerSec.tick(FPS)

print("Score : " + str(target_counter))
