import asyncio
import gymnasium as gym
import cv2 as cv2
import time
import pygame
import numpy as np
import os as os


async def main():
    env = gym.make('Blackjack-v1', render_mode='rgb_array')
    env.reset()
    Blackjack_Action_Map = {0:"Stick", 1:"Hit"}
    

    if os.path.isfile("tamer/Blackjack/Data/Blackjack_Data.npy"):
        Blackjack_Data = np.load("tamer/Blackjack/Data/Blackjack_Data.npy")
    else:
        Blackjack_Data = np.zeros(shape=(0,5))
        np.save("tamer/Blackjack/Data/Blackjack_Data.npy",Blackjack_Data)


    cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)

    for i in range(100):
        env.reset()
        observation_example = env.observation_space.sample()
        while observation_example[0]>21:
            observation_example = env.observation_space.sample()
        env.unwrapped.state = observation_example
        player_sum, dealer_showing_card_value, usable_ace = env.unwrapped.state
        env.render()
        frame_bgr = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        cv2.imshow('OpenAI Gymnasium Playing', frame_bgr)
        screen = pygame.display.set_mode((200, 100))
        area = screen.fill((0, 0, 0))
        pygame.display.update(area)
        font = pygame.font.Font("freesansbold.ttf", 32)
        action_id = int(np.random.random()*len(Blackjack_Action_Map))
        action = Blackjack_Action_Map[action_id]
        text = font.render(action, True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = screen.blit(text, text_rect)
        pygame.display.update(area)

        while True:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    area = screen.fill((0, 255, 0))
                    reward = 1
                    break
                elif event.key == pygame.K_b:
                    area = screen.fill((255, 0, 0))
                    reward = -1
                    break
        new_data = np.array((player_sum, dealer_showing_card_value, usable_ace,action_id,reward))[np.newaxis]
        Blackjack_Data = np.concatenate([Blackjack_Data,new_data])
    np.save("tamer/Blackjack/Data/Blackjack_Data.npy",Blackjack_Data)
        
if __name__ == '__main__':
    asyncio.run(main())


