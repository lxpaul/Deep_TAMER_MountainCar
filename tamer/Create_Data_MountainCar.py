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
    MountainCar_Action_Map = {0:"left", 1:"none", 2:"right"}
    
    cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)
    
    MountainCar_Data = np.zeros((10,4))

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
        action_id = int(np.random.random()*3)
        action = MountainCar_Action_Map[action_id]
        text = font.render(action, True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = screen.blit(text, text_rect)
        pygame.display.update(area)

        window_X = frame_bgr.shape[1]
        x_scale = window_X/1.8
        window_Y = frame_bgr.shape[0]
        y_scale = window_Y/2

        start_arrow = (int((pos+1.2) * x_scale),
                       int(window_Y - ((np.sin(3*pos) + 1) * y_scale)))
        slope = 3 * np.cos(3 * pos)
        angle = np.arctan(slope)
        arrow_length = 2000 * vel
        end_arrow = (int(start_arrow[0] + arrow_length * np.cos(angle)),
                     int(start_arrow[1] - arrow_length * np.sin(angle)))
        color = (128,0,128)
        thick = 2
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
        MountainCar_Data[i] = (pos,vel,action_id,reward)
        print(MountainCar_Data)
        




if __name__ == '__main__':
    asyncio.run(main())


