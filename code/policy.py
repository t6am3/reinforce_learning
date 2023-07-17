import numpy as np
import pandas as pd
import random
from utils import gen_random_prob


class Policy:
    def __init__(self, states_n, actions_set):
        # random init
        self.states_n = states_n
        self.actions_set = actions_set
        self.actions_n = len(self.actions_set)
        self.policy = np.array([gen_random_prob(actions_set) for _ in range(states_n)])

    def update_policy(self, action_values, epsilon=0):
        # set epsilon greedy policy, greedy policy is default
        explore_prob = epsilon / self.actions_n
        opt_prob = 1 - explore_prob * (self.actions_n - 1)
        new_policy = [[explore_prob for _ in self.actions_set] for _ in range(self.states_n)]
        opt_policy = np.argmax(action_values, axis=1)
        for row in range(len(opt_policy)):
            opt_p = opt_policy[row]
            new_policy[row][opt_p] = opt_prob
        
        del self.policy
        self.policy = new_policy

    def choose_action(self, state):
        dice = random.random()
        prob_seq = self.policy[state]
        for i in range(self.actions_n):
            action_prob = prob_seq[i]
            if dice <= action_prob:
                return i
            
            dice -= action_prob
        
        return self.actions_n - 1