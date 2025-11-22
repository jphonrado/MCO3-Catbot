import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################
import matplotlib.pyplot as plt







#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 17})

    alpha = 0.5            # Learning rate
    gamma = 0.9            # Discount factor
    epsilon = 1.0          # Amount of randomness in the action selection
    epsilon_decay = 0.001  # Fixed amount to decrease

    # Track outcomes
    outcomes = []

    print('Q-table before training:')
    print(q_table)
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
        
        state, info = env.reset()
        done = False

        outcomes.append("Failure")

        while not done:

            # Explore or exploit
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])


            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Q-learning update
            q_table[state][action] = q_table[state][action] + \
                alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state][action])

            state = new_state

            if reward != 0:
                outcomes[-1] = "Success"

        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay, 0)

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    print()
    print('===========================================')
    print('Q-table after training:')
    print(q_table)
    
    # Plot outcomes
    plt.figure(figsize=(12, 5))
    plt.xlabel("Run number")
    plt.ylabel("Outcome")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
    plt.show()

    return q_table