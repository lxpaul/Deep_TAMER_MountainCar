import asyncio
import gymnasium as gym
import cv2 as cv2
import time
import pygame
from H_Network import *
import numpy as np


async def main():

    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env.reset()
    
    cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)

    for i in range(10):
        observation_example = env.observation_space.sample()
        env.unwrapped.state = observation_example
        print(env.unwrapped.state)
        pos,vel = env.unwrapped.state
        env.render()

        frame_bgr = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        cv2.imshow('OpenAI Gymnasium Playing', frame_bgr)
        screen = pygame.display.set_mode((200, 100))
        area = screen.fill((0, 0, 0))
        pygame.display.update(area)
        font = pygame.font.Font("freesansbold.ttf", 32)
        action = "left"
        text = font.render(action, True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = screen.blit(text, text_rect)
        pygame.display.update(area)

        window_X = frame_bgr.shape[1]
        window_Y = frame_bgr.shape[0]

        start_arrow = (int((pos+1.2)/1.8 * window_X),int(np.cos(3*pos)*window_Y))
        end_arrow = (600,400)
        color = (255,0,0)
        thick = 10
        arrow = cv2.arrowedLine(frame_bgr, start_arrow, end_arrow, 
                                     color, thick)  
        cv2.imshow('OpenAI Gymnasium Playing', arrow)
        while True:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    area = screen.fill((0, 255, 0))
                    reward = 1
                    break
                elif event.key == pygame.K_a:
                    area = screen.fill((255, 0, 0))
                    reward = -1
                    break
        




if __name__ == '__main__':
    asyncio.run(main())


