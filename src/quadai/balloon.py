"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/Quadcopter-AI

This is the main game where you can compete with AI agents
Collect as many balloons within the time limit
"""
import os
from random import randrange
from math import sin, cos, pi, sqrt

import numpy as np
import pygame
from pygame.locals import *
from quadai.player import HumanPlayer, PIDPlayer, SACPlayer


def correct_path(current_path):
    """
    This function is used to get the correct path to the assets folder
    """
    return os.path.join(os.path.dirname(__file__), current_path)


def balloon():
    """
    Runs the balloon game.
    """
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
        image = pygame.image.load(
            correct_path(
                os.path.join(
                    "assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-"
                    + str(i)
                    + ".png"
                )
            )
        )
        image.convert()
        player_animation.append(
            pygame.transform.scale(image, (player_width, int(player_width * 0.30)))
        )

    target_width = 30
    target_animation_speed = 0.1
    target_animation = []
    for i in range(1, 8):
        image = pygame.image.load(
            correct_path(
                os.path.join(
                    "assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/red-plain-"
                    + str(i)
                    + ".png"
                )
            )
        )
        image.convert()
        target_animation.append(
            pygame.transform.scale(image, (target_width, int(target_width * 1.73)))
        )

    # Loading background sprites
    cloud1 = pygame.image.load(
        correct_path(
            os.path.join(
                "assets/balloon-flat-asset-pack/png/background-elements/cloud-1.png"
            )
        )
    )
    cloud2 = pygame.image.load(
        correct_path(
            os.path.join(
                "assets/balloon-flat-asset-pack/png/background-elements/cloud-2.png"
            )
        )
    )
    sun = pygame.image.load(
        correct_path(
            os.path.join(
                "assets/balloon-flat-asset-pack/png/background-elements/sun.png"
            )
        )
    )
    cloud1.set_alpha(124)
    (x_cloud1, y_cloud1, speed_cloud1) = (150, 200, 0.3)
    cloud2.set_alpha(124)
    (x_cloud2, y_cloud2, speed_cloud2) = (400, 500, -0.2)
    sun.set_alpha(124)

    # Loading fonts
    pygame.font.init()
    name_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Bold.ttf"), 20)
    name_hud_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Bold.ttf"), 15)
    time_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Bold.ttf"), 30)
    score_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Regular.ttf"), 20)
    respawn_timer_font = pygame.font.Font(
        correct_path("assets/fonts/Roboto-Bold.ttf"), 90
    )
    respawning_font = pygame.font.Font(
        correct_path("assets/fonts/Roboto-Regular.ttf"), 15
    )

    # Function to display info about a player

    def display_info(position):
        name_text = name_font.render(player.name, True, (255, 255, 255))
        screen.blit(name_text, (position, 20))
        target_text = score_font.render(
            "Score : " + str(player.target_counter), True, (255, 255, 255)
        )
        screen.blit(target_text, (position, 45))
        if player.dead == True:
            respawning_text = respawning_font.render(
                "Respawning...", True, (255, 255, 255)
            )
            screen.blit(respawning_text, (position, 70))

    # Initialize game variables
    time = 0
    step = 0
    time_limit = 100
    respawn_timer_max = 3

    players = [HumanPlayer(), PIDPlayer(), SACPlayer()]

    # Generate 100 targets
    targets = []
    for i in range(100):
        targets.append((randrange(200, 600), randrange(200, 600)))

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

        # For each player
        for player_index, player in enumerate(players):
            if player.dead == False:
                # Initialize accelerations
                player.x_acceleration = 0
                player.y_acceleration = gravity
                player.angular_acceleration = 0

                # Calculate propeller force in function of input
                if player.name == "DQN" or player.name == "PID":
                    thruster_left, thruster_right = player.act(
                        [
                            targets[player.target_counter][0] - player.x_position,
                            player.x_speed,
                            targets[player.target_counter][1] - player.y_position,
                            player.y_speed,
                            player.angle,
                            player.angular_speed,
                        ]
                    )
                elif player.name == "SAC":
                    angle_to_up = player.angle / 180 * pi
                    velocity = sqrt(player.x_speed**2 + player.y_speed**2)
                    angle_velocity = player.angular_speed
                    distance_to_target = (
                        sqrt(
                            (targets[player.target_counter][0] - player.x_position) ** 2
                            + (targets[player.target_counter][1] - player.y_position)
                            ** 2
                        )
                        / 500
                    )
                    angle_to_target = np.arctan2(
                        targets[player.target_counter][1] - player.y_position,
                        targets[player.target_counter][0] - player.x_position,
                    )
                    # Angle between the to_target vector and the velocity vector
                    angle_target_and_velocity = np.arctan2(
                        targets[player.target_counter][1] - player.y_position,
                        targets[player.target_counter][0] - player.x_position,
                    ) - np.arctan2(player.y_speed, player.x_speed)
                    distance_to_target = (
                        sqrt(
                            (targets[player.target_counter][0] - player.x_position) ** 2
                            + (targets[player.target_counter][1] - player.y_position)
                            ** 2
                        )
                        / 500
                    )
                    thruster_left, thruster_right = player.act(
                        np.array(
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
                    )
                else:
                    thruster_left, thruster_right = player.act([])

                # Calculate accelerations according to Newton's laws of motion
                player.x_acceleration += (
                    -(thruster_left + thruster_right)
                    * sin(player.angle * pi / 180)
                    / mass
                )
                player.y_acceleration += (
                    -(thruster_left + thruster_right)
                    * cos(player.angle * pi / 180)
                    / mass
                )
                player.angular_acceleration += (
                    arm * (thruster_right - thruster_left) / mass
                )

                # Calculate speed
                player.x_speed += player.x_acceleration
                player.y_speed += player.y_acceleration
                player.angular_speed += player.angular_acceleration

                # Calculate position
                player.x_position += player.x_speed
                player.y_position += player.y_speed
                player.angle += player.angular_speed

                # Calculate distance to target
                dist = sqrt(
                    (player.x_position - targets[player.target_counter][0]) ** 2
                    + (player.y_position - targets[player.target_counter][1]) ** 2
                )

                # If target reached, respawn target
                if dist < 50:
                    player.target_counter += 1

                # If to far, die and respawn after timer
                elif dist > 1000:
                    player.dead = True
                    player.respawn_timer = respawn_timer_max
            else:
                # Display respawn timer
                if player.name == "Human":
                    respawn_text = respawn_timer_font.render(
                        str(int(player.respawn_timer) + 1), True, (255, 255, 255)
                    )
                    respawn_text.set_alpha(124)
                    screen.blit(
                        respawn_text,
                        (
                            WIDTH / 2 - respawn_text.get_width() / 2,
                            HEIGHT / 2 - respawn_text.get_height() / 2,
                        ),
                    )

                player.respawn_timer -= 1 / 60
                # Respawn
                if player.respawn_timer < 0:
                    player.dead = False
                    (
                        player.angle,
                        player.angular_speed,
                        player.angular_acceleration,
                    ) = (
                        0,
                        0,
                        0,
                    )
                    (player.x_position, player.x_speed, player.x_acceleration) = (
                        400,
                        0,
                        0,
                    )
                    (player.y_position, player.y_speed, player.y_acceleration) = (
                        400,
                        0,
                        0,
                    )

            # Display target and player
            target_sprite = target_animation[
                int(step * target_animation_speed) % len(target_animation)
            ]
            target_sprite.set_alpha(player.alpha)
            screen.blit(
                target_sprite,
                (
                    targets[player.target_counter][0]
                    - int(target_sprite.get_width() / 2),
                    targets[player.target_counter][1]
                    - int(target_sprite.get_height() / 2),
                ),
            )

            player_sprite = player_animation[
                int(step * player_animation_speed) % len(player_animation)
            ]
            player_copy = pygame.transform.rotate(player_sprite, player.angle)
            player_copy.set_alpha(player.alpha)
            screen.blit(
                player_copy,
                (
                    player.x_position - int(player_copy.get_width() / 2),
                    player.y_position - int(player_copy.get_height() / 2),
                ),
            )

            # Display player name
            name_hud_text = name_hud_font.render(player.name, True, (255, 255, 255))
            screen.blit(
                name_hud_text,
                (
                    player.x_position - int(name_hud_text.get_width() / 2),
                    player.y_position - 30 - int(name_hud_text.get_height() / 2),
                ),
            )

            # Display player info
            if player_index == 0:
                display_info(20)
            elif player_index == 1:
                display_info(130)
            elif player_index == 2:
                display_info(240)
            elif player_index == 3:
                display_info(350)

            time_text = time_font.render(
                "Time : " + str(int(time_limit - time)), True, (255, 255, 255)
            )
            screen.blit(time_text, (670, 30))

        # Ending conditions
        if time > time_limit:
            break

        pygame.display.update()
        FramePerSec.tick(FPS)

    # Print scores and who won
    print("")
    scores = []
    for player in players:
        print(player.name + " collected : " + str(player.target_counter))
        scores.append(player.target_counter)
    winner = players[np.argmax(scores)].name

    print("")
    print("Winner is : " + winner + " !")
