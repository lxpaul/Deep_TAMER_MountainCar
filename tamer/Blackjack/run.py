"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
# import gym
import gymnasium as gym
import matplotlib as plt

from agent import Tamer


async def main(model:str):
    env = gym.make('Blackjack-v1', render_mode='rgb_array')
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 1000
    tame = True  # set to false for vanilla Q learning

    tamer_training_timestep = 0.4
    
    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=model)
    
    print("Awaiting agent.train")
    
    await agent.train()
    
    print("Starting agent.play")
    agent.play(n_episodes=1, render=False)
    agent.evaluate(n_episodes=1000)

if __name__ == '__main__':
    model_name = input("model name: ")
    asyncio.run(main(model_name))
