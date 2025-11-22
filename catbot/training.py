#REFERENCES
#Labonne, M. (2022, March 7). Q‑learning for beginners: Train an AI to solve the Frozen Lake environment. Medium. https://medium.com/data-science/q-learning-for-beginners-2837b777741

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

# Helper: decode observation integer into (agent_r, agent_c, cat_r, cat_c)
#decode_obs converts the single integer observation returned by CatChaseEnv 
# into the four coordinates (agent_row, agent_col, cat_row, cat_col). 
# That allows the training code to compute distances, 
# detect movement/wall bumps, and shape rewards. 
def decode_obs(obs: int):
    obs = int(obs)
    agent_r = obs // 1000
    rem = obs % 1000
    agent_c = rem // 100
    rem = rem % 100
    cat_r = rem // 10
    cat_c = rem % 10
    return agent_r, agent_c, cat_r, cat_c
#############################################################################G
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

    alpha = 0.5            # Learning rate
    gamma = 0.9            # Discount factor
    epsilon = 1.0          # Amount of randomness in the action selection
    epsilon_decay = 0.001  # Fixed amount to decrease

    # Track outcomes
    outcomes = []

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
        state = int(state)
        done = False

        outcomes.append("Failure")

        while not done:

            # Explore or exploit
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            action = int(action)

            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = int(new_state)
            done = terminated or truncated

            # Compute shaped reward based on Manhattan distance between agent and cat
            #agent_row(ar), agent_col(ac), cat_row(cr), cat_col(cc)
            ar, ac, cr, cc = decode_obs(state)
            nar, nac, ncr, ncc = decode_obs(new_state) #new state variables

            prev_dist = abs(ar - cr) + abs(ac - cc)
            new_dist = abs(nar - ncr) + abs(nac - ncc)

            # Default shaped reward: small step penalty, bonus for moving closer,
            # penalty for moving away, penalty for bumping into walls, success reward.
            reward = -0.01
            if new_dist < prev_dist:
                reward += 0.05
            elif new_dist > prev_dist:
                reward -= 0.02

            # If agent didn't move (bumped into wall), penalize slightly
            agent_moved = (nar != ar) or (nac != ac)
            if not agent_moved:
                reward -= 0.02

            # Success reward on catch
            if new_dist == 0 and done:
                reward = 1.0

            # Q-learning update using computed reward
            q_table[state][action] = q_table[state][action] + \
                alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state][action])

            state = new_state

            if done and new_dist == 0:
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

    return q_table

