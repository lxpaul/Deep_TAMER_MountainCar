import datetime as dt
import os
import pickle
import time
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter

import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import torch as th
from H_Network import *

MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

import matplotlib.pyplot as plt
import cv2

class SGDFunctionApproximator:
    """ SGD function approximator with RBF preprocessing. """
    def __init__(self, env):
        
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        )

        print(observation_examples[0, :])
        # Standardize features by removing the mean and scaling to unit variance.
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples) # Compute mean and variance

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))
        # print(self.scaler.transform(observation_examples).shape)
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """ Returns the featurized representation for a state. """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

class Tamer:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,  # only affects Q-learning
        epsilon=0, # only affects Q-learning
        min_eps=0,  # minimum value for epsilon after annealing
        tame=True,  # set to false for normal Q-learning
        ts_len=0.2,  # length of timestep for training TAMER
        output_dir=LOGS_DIR,
        model_file_to_load=None,  # filename of pretrained model
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.human_reward_model = H_prediction_model(3)
        self.human_reward_model.load_state_dict(th.load("tamer/saved_models/H_model_first_try", weights_only=True))
        self.human_reward_model.eval()

        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if self.tame:
                self.H = SGDFunctionApproximator(env)  # init H function
            else:  # optionally run as standard Q Learning
                self.Q = SGDFunctionApproximator(env)  # init Q function

        # hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps
        self.reward_factor = 1.0      # A factor to multiply the reward by
        # calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # reward logging
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
            'Combined Reward',
        ]

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index, disp):
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        # print(f'Episode: {episode_index + 1}  Timestep:')
        cv2.namedWindow('OpenAI Gymnasium Training', cv2.WINDOW_NORMAL)

        rng = np.random.default_rng()
        tot_reward = 0
        state, _ = self.env.reset()

        for ts in count():
            print(f' {ts}', end='')
            # print(f' {ts}')
            # self.env.render()
            frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
            # print(self.env.render())
            cv2.imshow('OpenAI Gymnasium Training', frame_bgr)
            key = cv2.waitKey(25)  # Adjust the delay (25 milliseconds in this case)
            if key == 27:
                break

            # Determine next action
            action = self.act(state)
            if self.tame:
                disp.show_action(action)

            # Get next state and reward
            next_state, reward, done, info, _ = self.env.step(action)

            if not self.tame:
                if done and next_state[0] >= 0.5:
                    td_target = reward
                else:
                    td_target = reward + self.discount_factor * np.max(
                        self.Q.predict(next_state)
                    )
                self.Q.update(state, action, td_target)
            else:
                now = time.time()
                while time.time() < now + self.ts_len:
                    frame = None
                    time.sleep(0.01)  # save the CPU
                    human_reward = th.sign(self.human_reward_model(th.tensor([state[0],state[1],action],dtype=th.float32))).item()

                    if human_reward != 0:
                        
                        self.H.update(state, action, human_reward)

                        break

            tot_reward += reward
            if done:
                print(f'  Reward: {tot_reward}')
                cv2.destroyAllWindows()
                break


            state = next_state

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    async def train(self, model_file_to_save=None):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """
        # render first so that pygame display shows up on top
        # self.env.render()
        
        disp = None
        if self.tame:
            # only init pygame display if we're actually training tamer
            from interface import Interface
            disp = Interface(action_map=MOUNTAINCAR_ACTION_MAP)
            #disp = Interface(action_map=LUNARLANDER_ACTION_MAP)
            #disp = Interface(action_map=CARTPOLE_ACTION_MAP)

        for i in range(self.num_episodes):
            print(f"Num episode : {i}")
            self._train_episode(i, disp)

        print('\nCleaning up...')
        self.env.close()

    def play(self, n_episodes=1, render=False):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        if render:            
            cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)

        self.epsilon = 0
        ep_rewards = []
        for i in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            tot_reward = 0
            # TODO setup a duration criterion in case of impossibility to find a solution
            while not done:
                action = self.act(state)
                next_state, reward, done, info, _ = self.env.step(action)
                tot_reward += reward
                if render:
                    # self.env.render()
                    frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.imshow('OpenAI Gymnasium Playing', frame_bgr)
                    key = cv2.waitKey(25)  # Adjust the delay (25 milliseconds in this case)
                    if key == 27:
                        break
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        self.env.close()
        if render:
            cv2.destroyAllWindows()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        return avg_reward
