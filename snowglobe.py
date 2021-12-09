"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/2D-Quadcopter-AI

Christmas themed mod to the main game
"""

import numpy as np
import pygame
import os
from pygame.locals import *
from math import sin, cos, pi, sqrt, atan2
from random import randrange
from player import HumanPlayer, PIDPlayer, DQNPlayer

# Game constants
FPS = 60
WIDTH = 800
HEIGHT = 800

snowglobe_radius = 350

# Physics constants
gravity = 0.08
# Propeller force for UP and DOWN
thruster_amplitude = 0.04
# Propeller force for LEFT and RIGHT rotations
diff_amplitude = 0.003
# By default, thruster will apply angle force of thruster_mean
thruster_mean = 0.04
mass = 1
# Length from center of mass to propeller
arm = 25

# Initialize Pygame, load sprites
FramePerSec = pygame.time.Clock()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Loading player and target sprites
player_width = 80
player_animation_speed = 0.3
player_animation = []
for i in range(1, 5):
    image = pygame.image.load(os.path.join(
        "assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-" + str(i) + ".png"))
    image.convert()
    player_animation.append(pygame.transform.scale(
        image, (player_width, int(player_width*0.30))))

# Loading background sprites
cloud1 = pygame.image.load(os.path.join(
    "assets/balloon-flat-asset-pack/png/background-elements/cloud-1.png"))
cloud2 = pygame.image.load(os.path.join(
    "assets/balloon-flat-asset-pack/png/background-elements/cloud-2.png"))
sun = pygame.image.load(os.path.join(
    "assets/balloon-flat-asset-pack/png/background-elements/sun.png"))
cloud1.set_alpha(124)
(x_cloud1, y_cloud1, speed_cloud1) = (150, 200, 0.3)
cloud2.set_alpha(124)
(x_cloud2, y_cloud2, speed_cloud2) = (400, 500, -0.2)
sun.set_alpha(124)


# Initialize game variables
time = 0
step = 0
time_limit = 30
respawn_timer_max = 3

player = PIDPlayer()


class SnowParticle:
    def __init__(self, x, y, radius):
        self.x_position = x
        self.y_position = y
        self.x_speed = 0
        self.y_speed = 0
        self.x_acceleration = 0
        self.y_acceleration = 0
        self.radius = radius


snow_particles = []
n_particles = 2000
for i in range(0, n_particles):
    snow_particles.append(SnowParticle(
        randrange(150, WIDTH - 150), randrange(150, HEIGHT-150), randrange(1, 3)))

boost = False


# Game loop
while True:
    pygame.event.get()

    boost = False
    # If spacebar is pressed, boost
    if pygame.key.get_pressed()[K_SPACE]:
        boost = True

    # Display background
    screen.fill((131, 176, 181))

    x_cloud1 += speed_cloud1
    if x_cloud1 > WIDTH:
        x_cloud1 = -cloud1.get_width()
    screen.blit(cloud1, (x_cloud1, y_cloud1))

    x_cloud2 += speed_cloud2
    if x_cloud2 < -cloud2.get_width():
        x_cloud2 = WIDTH
    screen.blit(cloud2, (x_cloud2, y_cloud2))

    screen.blit(sun, (630, -100))

    time += 1/60
    step += 1

    # Initialize accelerations
    player.x_acceleration = 0
    player.y_acceleration = gravity
    player.angular_acceleration = 0

    # Get mouse position
    mouse_pos = pygame.mouse.get_pos()
    (target_x, target_y) = mouse_pos

    # Calculate propeller force in function of input
    thruster_left, thruster_right = player.act(
        [target_x - player.x_position, player.x_speed, target_y - player.y_position, player.y_speed, player.angle, player.angular_speed])

    # Calculate accelerations according to Newton's laws of motion
    force_x = -(thruster_left + thruster_right) * \
        sin(player.angle*pi/180)/mass
    force_y = -(thruster_left + thruster_right) * \
        cos(player.angle*pi/180)/mass

    position = [player.x_position, player.y_position]
    force = [force_x, force_y]

    distance_to_center = sqrt((position[0] - WIDTH/2)**2 +
                              (position[1] - HEIGHT/2)**2)

    # Update accelerations
    player.x_acceleration += force[0]
    player.y_acceleration += force[1]

    player.angular_acceleration += arm * \
        (thruster_right - thruster_left)/mass

    # Calculate speed
    player.x_speed += player.x_acceleration
    player.y_speed += player.y_acceleration
    player.angular_speed += player.angular_acceleration

    angle = atan2(position[1] - HEIGHT/2, position[0] - WIDTH/2)
    r_speed = player.x_speed * cos(angle) + player.y_speed * sin(angle)

    if distance_to_center > snowglobe_radius*0.9 and r_speed > 0:
        theta_speed = player.x_speed * \
            sin(angle) - player.y_speed * cos(angle)
        r_speed = - r_speed*0.8
        x_speed = r_speed * cos(angle) + theta_speed * sin(angle)
        y_speed = r_speed * sin(angle) - theta_speed * cos(angle)
        player.x_speed = x_speed
        player.y_speed = y_speed

    # Calculate position
    player.x_position += player.x_speed
    player.y_position += player.y_speed
    player.angle += player.angular_speed

    player_sprite = player_animation[int(
        step*player_animation_speed) % len(player_animation)]
    player_copy = pygame.transform.rotate(player_sprite, player.angle)
    player_copy.set_alpha(255)
    screen.blit(player_copy, (player.x_position - int(player_copy.get_width()/2),
                player.y_position - int(player_copy.get_height()/2)))

    # Draw a ring in the center of the screen
    pygame.draw.circle(screen, (200, 200, 200),
                       (int(WIDTH/2), int(HEIGHT/2)), snowglobe_radius, 5)

    for snow_particle in snow_particles:
        snow_particle.x_acceleration = 0
        snow_particle.y_acceleration = gravity*0.2

        distance_to_player = sqrt((snow_particle.x_position - player.x_position)**2 +
                                  (snow_particle.y_position - player.y_position)**2)

        if distance_to_player < 100:
            snow_particle.x_acceleration += 2*(
                snow_particle.x_position - player.x_position)/distance_to_player
            snow_particle.y_acceleration += 2*(
                snow_particle.y_position - player.y_position)/distance_to_player

        snow_particle.x_speed += snow_particle.x_acceleration - 0.02*snow_particle.x_speed
        snow_particle.y_speed += snow_particle.y_acceleration - 0.02*snow_particle.y_speed

        position = [snow_particle.x_position, snow_particle.y_position]
        distance_to_center = sqrt((position[0] - WIDTH/2)**2 +
                                  (position[1] - HEIGHT/2)**2)

        angle = atan2(position[1] - HEIGHT/2, position[0] - WIDTH/2)
        r_speed = snow_particle.x_speed * \
            cos(angle) + snow_particle.y_speed * sin(angle)
        theta_speed = snow_particle.x_speed * \
            sin(angle) - snow_particle.y_speed * cos(angle)
        theta_position = snow_particle.x_position * \
            sin(angle) - snow_particle.y_position * cos(angle)

        if distance_to_center > snowglobe_radius*1 and r_speed > 0:
            r_speed = - r_speed*0.7
            theta_speed = theta_speed
            x_speed = r_speed * cos(angle) + theta_speed * sin(angle)
            y_speed = r_speed * sin(angle) - theta_speed * cos(angle)
            snow_particle.x_speed = x_speed
            snow_particle.y_speed = y_speed

        if boost:
            snow_particle.y_speed -= 1
        snow_particle.x_position += snow_particle.x_speed
        snow_particle.y_position += snow_particle.y_speed

        pygame.draw.circle(screen, (255, 255, 255), (int(snow_particle.x_position), int(
            snow_particle.y_position)), snow_particle.radius, 0)

    pygame.display.update()
    FramePerSec.tick(FPS)
