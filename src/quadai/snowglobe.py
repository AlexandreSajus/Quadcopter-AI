"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/Quadcopter-AI

Christmas themed mod to the main game
"""

import os
import random
from random import randrange
from math import sin, cos, pi, sqrt, atan2

import pygame
from pygame.locals import *

from quadai.player import PIDPlayer


def correct_path(current_path):
    """
    This function is used to get the correct path to the assets folder
    """
    return os.path.join(os.path.dirname(__file__), current_path)


def snowglobe():
    """
    Runs the snowglobe game
    """
    # FPS, w and h of the window
    FPS = 60
    WIDTH = 800
    HEIGHT = 800

    # simulation step number
    step = 0
    # snowglobe geometry
    snowglobe_radius = 350
    snowglobe_edge = 20
    # radius of the snow particles
    (snow_min_radius, snow_max_radius) = (3, 9)
    # number of snow particles
    n_particles = 1800
    # factor to define collision player-snowglobe
    player_collision_margin = 0.9
    # factor to define collision player-snow
    particle_collision_margin = 0.95
    # drag coefficient for snow
    drag = 0.02
    # gravity reduction for snow
    gravity_reduction = 0.2
    # snow-drone interaction force
    interaction_force_x = 0.1
    interaction_force_y = 0.08
    # snow-drone interaction distance
    interaction_distance = 150
    # snow-drone interaction cone angle
    interaction_angle = 30
    # random snow speed
    random_snow_speed = 0.1
    # snow collision drag
    collision_drag = 0.3

    # Drone constants
    # Physics constants
    gravity = 0.08
    # Mass
    mass = 1
    # Length from center of mass to propeller
    arm = 25

    # Initialize Pygame, load sprites
    FramePerSec = pygame.time.Clock()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Title the window
    pygame.display.set_caption("Schneekugel")

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

    # Init drone and snow particles
    # Create player
    player = PIDPlayer()

    class SnowParticle:
        """
        A snow particle
        """

        def __init__(self, x, y, radius):
            self.x_position = x
            self.y_position = y
            self.x_speed = 0
            self.y_speed = 0
            self.x_acceleration = 0
            self.y_acceleration = 0
            self.radius = radius

    # Utilities
    def create_snow_particles(n_particles):
        snow_particles = []
        for _ in range(0, n_particles):
            while True:
                x = randrange(0, WIDTH)
                y = randrange(0, HEIGHT)
                if (
                    sqrt((x - WIDTH / 2) ** 2 + (y - HEIGHT / 2) ** 2)
                    < snowglobe_radius
                ):
                    break
            r = randrange(snow_min_radius, snow_max_radius)
            snow_particles.append(SnowParticle(x, y, r))
        return snow_particles

    def convert_to_circular(x, y, x_pos, y_pos):
        """
        Convert cartesian coordinates to circular
        Returns r and theta coordinates
        """
        # Because pygame y-axis is inverted, y = -y
        angle = atan2(HEIGHT / 2 - y_pos, x_pos - WIDTH / 2)
        y = -y
        r = x * cos(angle) + y * sin(angle)
        theta = -sin(angle) * x + cos(angle) * y
        return r, theta

    def convert_to_cartesian(r, theta, x_pos, y_pos):
        """
        Convert circular coordinates to cartesian
        Returns x and y coordinates
        """
        # Because pygame y-axis is inverted, y = -y
        angle = atan2(HEIGHT / 2 - y_pos, x_pos - WIDTH / 2)
        x = r * cos(angle) - theta * sin(angle)
        y = r * sin(angle) + theta * cos(angle)
        y = -y
        return x, y

    snow_particles = create_snow_particles(n_particles)

    # Game loop
    while True:

        # Quit if user closes window
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

        step += 1

        # If spacebar is pressed, reset snow particles
        if pygame.key.get_pressed()[pygame.K_SPACE]:
            snow_particles = create_snow_particles(n_particles)

        # Background
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

        # Snowglobe
        pygame.draw.circle(
            screen,
            (200, 200, 200),
            (int(WIDTH / 2), int(HEIGHT / 2)),
            snowglobe_radius,
            snowglobe_edge,
        )

        # Player
        # Initialize accelerations
        player.x_acceleration = 0
        player.y_acceleration = gravity
        player.angular_acceleration = 0

        # Get mouse position
        mouse_pos = pygame.mouse.get_pos()
        (target_x, target_y) = mouse_pos

        # Calculate propeller force in function of input
        thruster_left, thruster_right = player.act(
            [
                target_x - player.x_position,
                player.x_speed,
                target_y - player.y_position,
                player.y_speed,
                player.angle,
                player.angular_speed,
            ]
        )

        # Calculate accelerations according to Newton's laws of motion
        force_x = (
            -(thruster_left + thruster_right) * sin(player.angle * pi / 180) / mass
        )
        force_y = (
            -(thruster_left + thruster_right) * cos(player.angle * pi / 180) / mass
        )

        # Update accelerations
        player.x_acceleration += force_x
        player.y_acceleration += force_y

        player.angular_acceleration += arm * (thruster_right - thruster_left) / mass

        # Calculate speed
        player.x_speed += player.x_acceleration
        player.y_speed += player.y_acceleration
        player.angular_speed += player.angular_acceleration

        # Boundary conditions
        distance_to_center = sqrt(
            (player.x_position - WIDTH / 2) ** 2 + (player.y_position - HEIGHT / 2) ** 2
        )
        r_speed, theta_speed = convert_to_circular(
            player.x_speed, player.y_speed, player.x_position, player.y_position
        )

        if (
            distance_to_center > snowglobe_radius * player_collision_margin
            and r_speed > 0
        ):
            r_speed = -r_speed
            x_speed, y_speed = convert_to_cartesian(
                r_speed, theta_speed, player.x_position, player.y_position
            )
            player.x_speed = x_speed
            player.y_speed = y_speed

        # Calculate position
        player.x_position += player.x_speed
        player.y_position += player.y_speed
        player.angle += player.angular_speed

        # Animation
        player_sprite = player_animation[
            int(step * player_animation_speed) % len(player_animation)
        ]
        player_copy = pygame.transform.rotate(player_sprite, player.angle)
        player_copy.set_alpha(255)
        screen.blit(
            player_copy,
            (
                player.x_position - int(player_copy.get_width() / 2),
                player.y_position - int(player_copy.get_height() / 2),
            ),
        )

        # Snow particles
        for snow_particle in snow_particles:
            # Accelerations
            snow_particle.x_acceleration = 0 - drag * snow_particle.x_speed
            snow_particle.y_acceleration = (
                gravity * gravity_reduction - drag * snow_particle.y_speed
            )

            # Drone interaction
            distance_to_player = sqrt(
                (snow_particle.y_position - player.y_position) ** 2
                + (snow_particle.x_position - player.x_position) ** 2
            )
            angle_to_player = atan2(
                snow_particle.y_position - player.y_position,
                player.x_position - snow_particle.x_position,
            )
            angle_to_player = angle_to_player * 180 / pi

            if distance_to_player < interaction_distance:
                # If snow below the player, snow is attracted to the player
                if interaction_angle < angle_to_player < 180 - interaction_angle:
                    snow_particle.x_acceleration -= (
                        interaction_force_x
                        * (snow_particle.x_position - player.x_position)
                        / distance_to_player
                    )
                    snow_particle.y_acceleration -= (
                        interaction_force_y
                        * (snow_particle.y_position - player.y_position)
                        / distance_to_player
                    )
                # If snow above the player, snow is repelled from the player
                elif -interaction_angle > angle_to_player > -180 + interaction_angle:
                    snow_particle.x_acceleration += (
                        interaction_force_x
                        * (snow_particle.x_position - player.x_position)
                        / distance_to_player
                    )
                    snow_particle.y_acceleration += (
                        interaction_force_y
                        * (snow_particle.y_position - player.y_position)
                        / distance_to_player
                    )

            # Calculate speed
            snow_particle.x_speed += snow_particle.x_acceleration
            snow_particle.y_speed += snow_particle.y_acceleration

            # Add random noise to speed
            snow_particle.x_speed += random.uniform(
                -random_snow_speed, random_snow_speed
            )
            snow_particle.y_speed += random.uniform(
                -random_snow_speed, random_snow_speed
            )

            # Boundary conditions
            distance_to_center = sqrt(
                (snow_particle.x_position - WIDTH / 2) ** 2
                + (snow_particle.y_position - HEIGHT / 2) ** 2
            )
            r_speed, theta_speed = convert_to_circular(
                snow_particle.x_speed,
                snow_particle.y_speed,
                snow_particle.x_position,
                snow_particle.y_position,
            )

            if (
                distance_to_center > snowglobe_radius * particle_collision_margin
                and r_speed > 0
            ):
                r_speed = -r_speed * collision_drag
                theta_speed = theta_speed * collision_drag
                x_speed, y_speed = convert_to_cartesian(
                    r_speed,
                    theta_speed,
                    snow_particle.x_position,
                    snow_particle.y_position,
                )
                snow_particle.x_speed = x_speed
                snow_particle.y_speed = y_speed

            # Calculate position
            snow_particle.x_position += snow_particle.x_speed
            snow_particle.y_position += snow_particle.y_speed

            # Clamp position
            r_position, theta_position = convert_to_circular(
                snow_particle.x_position - WIDTH / 2,
                snow_particle.y_position - HEIGHT / 2,
                snow_particle.x_position,
                snow_particle.y_position,
            )

            if r_position > snowglobe_radius * particle_collision_margin:
                r_position = snowglobe_radius * particle_collision_margin
                x_position, y_position = convert_to_cartesian(
                    r_position,
                    theta_position,
                    snow_particle.x_position,
                    snow_particle.y_position,
                )
                snow_particle.x_position = x_position + WIDTH / 2
                snow_particle.y_position = y_position + HEIGHT / 2

            # Animation
            pygame.draw.circle(
                screen,
                (255, 255, 255),
                (int(snow_particle.x_position), int(snow_particle.y_position)),
                snow_particle.radius,
                0,
            )

        pygame.display.update()
        FramePerSec.tick(FPS)
