import pygame
import sys
from pygame.locals import *

pygame.init()
vec = pygame.math.Vector2  # 2 for two dimensional

HEIGHT = 800
WIDTH = 800
FPS = 60
gravity = 0.2
thruster = 0.4

FramePerSec = pygame.time.Clock()

displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game")


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.surf = pygame.Surface((80, 30))
        self.surf.fill((128, 255, 40))
        self.rect = self.surf.get_rect(center=(200, 200))

        self.pos = vec((200, 200))
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)

    def move(self):
        self.acc = vec(0, 0)

        pressed_keys = pygame.key.get_pressed()

        if pressed_keys[K_UP]:
            self.acc.y += -thruster

        self.acc.y += gravity
        self.vel += self.acc
        self.pos += self.vel

        self.rect.midbottom = self.pos


class platform(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.surf = pygame.Surface((WIDTH, 20))
        self.surf.fill((255, 0, 0))
        self.rect = self.surf.get_rect(center=(WIDTH/2, HEIGHT - 10))


P1 = Player()

all_sprites = pygame.sprite.Group()
all_sprites.add(P1)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    P1.move()

    displaysurface.fill((0, 0, 0))

    for entity in all_sprites:
        displaysurface.blit(entity.surf, entity.rect)

    pygame.display.update()
    FramePerSec.tick(FPS)
