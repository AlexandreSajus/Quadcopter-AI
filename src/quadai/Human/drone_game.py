"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/Quadcopter-AI

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

# Loading player and target sprites
player_width = 80
player_animation_speed = 0.3
player = []
for i in range(1, 5):
    image = pygame.image.load(
        os.path.join(
            "assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-"
            + str(i)
            + ".png"
        )
    )
    image.convert()
    player.append(
        pygame.transform.scale(image, (player_width, int(player_width * 0.30)))
    )

target_width = 30
target_animation_speed = 0.1
target = []
for i in range(1, 8):
    image = pygame.image.load(
        os.path.join(
            "assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/red-plain-"
            + str(i)
            + ".png"
        )
    )
    image.convert()
    target.append(
        pygame.transform.scale(image, (target_width, int(target_width * 1.73)))
    )

# Loading background sprites
cloud1 = pygame.image.load(
    os.path.join("assets/balloon-flat-asset-pack/png/background-elements/cloud-1.png")
)
cloud2 = pygame.image.load(
    os.path.join("assets/balloon-flat-asset-pack/png/background-elements/cloud-2.png")
)
sun = pygame.image.load(
    os.path.join("assets/balloon-flat-asset-pack/png/background-elements/sun.png")
)
cloud1.set_alpha(124)
(x_cloud1, y_cloud1, speed_cloud1) = (150, 200, 0.3)
cloud2.set_alpha(124)
(x_cloud2, y_cloud2, speed_cloud2) = (400, 500, -0.2)
sun.set_alpha(124)

# Loading fonts
pygame.font.init()
info_font = pygame.font.Font("assets/fonts/Roboto-Regular.ttf", 30)
respawn_font = pygame.font.Font("assets/fonts/Roboto-Bold.ttf", 90)

# Initialize physics variables
(angle, angular_speed, angular_acceleration) = (0, 0, 0)
(x_position, x_speed, x_acceleration) = (400, 0, 0)
(y_position, y_speed, y_acceleration) = (400, 0, 0)
x_target = randrange(200, 600)
y_target = randrange(200, 600)

# Initialize game variables
target_counter = 0
time = 0
step = 0
time_limit = 30
dead = False
respawn_timer_max = 3
respawn_timer = 3

# Game loop
while True:
    pygame.event.get()

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

    time += 1 / 60
    step += 1

    if dead == False:
        # Initialize accelerations
        x_acceleration = 0
        y_acceleration = gravity
        angular_acceleration = 0

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
        x_acceleration += (
            -(thruster_left + thruster_right) * sin(angle * pi / 180) / mass
        )
        y_acceleration += (
            -(thruster_left + thruster_right) * cos(angle * pi / 180) / mass
        )
        angular_acceleration += arm * (thruster_right - thruster_left) / mass

        # Calculate speed
        x_speed += x_acceleration
        y_speed += y_acceleration
        angular_speed += angular_acceleration

        # Calculate position
        x_position += x_speed
        y_position += y_speed
        angle += angular_speed

        # Calculate distance to target
        dist = sqrt((x_position - x_target) ** 2 + (y_position - y_target) ** 2)

        # If target reached, respawn target
        if dist < 50:
            x_target = randrange(200, 600)
            y_target = randrange(200, 600)
            target_counter += 1

        # If to far, die and respawn after timer
        elif dist > 1000:
            dead = True
            respawn_timer = respawn_timer_max
    else:
        # Display respawn timer
        respawn_text = respawn_font.render(
            str(int(respawn_timer) + 1), True, (255, 255, 255)
        )
        respawn_text.set_alpha(124)
        screen.blit(
            respawn_text,
            (
                WIDTH / 2 - respawn_text.get_width() / 2,
                HEIGHT / 2 - respawn_text.get_height() / 2,
            ),
        )

        respawn_timer -= 1 / 60
        # Respawn
        if respawn_timer < 0:
            dead = False
            (angle, angular_speed, angular_acceleration) = (0, 0, 0)
            (x_position, x_speed, x_acceleration) = (400, 0, 0)
            (y_position, y_speed, y_acceleration) = (400, 0, 0)

    # Ending conditions
    if time > time_limit:
        break

    # Display target and player
    target_sprite = target[int(step * target_animation_speed) % len(target)]
    screen.blit(
        target_sprite,
        (
            x_target - int(target_sprite.get_width() / 2),
            y_target - int(target_sprite.get_height() / 2),
        ),
    )

    player_sprite = player[int(step * player_animation_speed) % len(player)]
    player_copy = pygame.transform.rotate(player_sprite, angle)
    screen.blit(
        player_copy,
        (
            x_position - int(player_copy.get_width() / 2),
            y_position - int(player_copy.get_height() / 2),
        ),
    )

    # Update text
    target_target = info_font.render(
        "Collected : " + str(target_counter), True, (255, 255, 255)
    )
    screen.blit(target_target, (20, 20))
    time_text = info_font.render(
        "Time : " + str(int(time_limit - time)), True, (255, 255, 255)
    )
    screen.blit(time_text, (20, 60))

    pygame.display.update()
    FramePerSec.tick(FPS)

print("Score : " + str(target_counter))
