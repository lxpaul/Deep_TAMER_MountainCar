# Deep TAMER Project

This repo is our final project for the human-machine interaction course.

Here we try to implement A version of Deep-TAMER based on the paper : Warnell, G., Waytowich, N.R., Lawhern, V.J., & Stone, P. (2017). Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces. ArXiv, abs/1709.10163.

## Principle

The idea can be summarized easily: we use the TAMER agent but instead of having human interaction to train it, it is a Deep neural-network trained to predict the human reward to any state.

## Deep Model and its training

Since we will apply our method to simple environments, our neural network will be simple as well.

Creation of the data is simple, we create states of our environment and ask a human user to tell if the action taken is good or bad (human reward). It can be done easily using the Create_Data file.

We then train our model with that data so that it can rpedict the human reward. This is done via the Train file.

What is interesting here is that, when the environment has different strategies, we can create different data sets and models for each strategies.

## MountainCar

The first environment we try is the MountainCar. Most of the code was already written, we swap the human reward in the training for our neural network's prediction.

## Blackjack

An environment with strategies that stays simple is the Blackjack environment (https://gymnasium.farama.org/environments/toy_text/blackjack/)

We create here three type of data and model for 3 strategies, Reckless, Normal and Cautious. Each are of varying degrees of cautiousness when deciding to hit or stand.