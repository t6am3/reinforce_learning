import numpy as np
import pandas as pd
import random
from copy import deepcopy
from utils import gen_random_prob
from maze import Maze
import model_config as mc
from policy import Policy

'''
value iteration for model-based
@input:
 1. maze
 2. reward - for any state and any action, give a reward, in the maze problem reward only depends on next state
 3. gamma - discounted rate
'''
class ValueIter:
    def __init__(self, maze, eps_stop=mc.eps_stop, eps_greedy=mc.eps_greedy, state_value_default=0, action_value_default=0., gamma=mc.gamma, alpha=mc.alpha):
        self.eps_stop = eps_stop
        self.eps_greedy = eps_greedy
        self.gamma = gamma
        self.alpha = alpha
        self.maze = maze
        self.state_values = np.array([state_value_default for _ in range(len(self.maze))])
        
        self.action_values = np.array([[action_value_default for _ in self.maze._actions_set] for _ in range(len(self.maze))])

        self.policy = Policy(len(self.maze), self.maze._actions_set)

        self.next_states = np.array([[maze.get_next_state(state, action) \
         for action in self.maze._actions_set] for state in range(len(self.maze))])

        self.reward = np.array([[maze.get_reward(state, self.maze._actions_set[i]) \
         for i in range(len(self.maze._actions_set))] for state in range(len(self.maze))])

    def print_state_df(self, array):
        print(pd.DataFrame(np.reshape(array, (self.maze.rows, self.maze.cols))))

    def print_df(self, array):
        df = pd.DataFrame(array)
        df.columns = self.maze._actions_set
        print(df)

    def print_state_values(self):
        print("State values:")
        self.print_state_df(self.state_values)

    def print_policy(self):
        print("Policy:")
        self.print_df(self.policy.policy)

    def print_action_values(self):
        print("Action values:")
        self.print_df(self.action_values)

    def print_reward(self):
        print("Reward:")
        self.print_df(self.reward)

    def print_next_states(self):
        print("Next states:")
        self.print_df(self.next_states)

    def print(self):
        self.maze.print()
        print()
        self.print_state_values()
        print()
        self.print_policy()
        print()
        self.print_action_values()
        print()
        self.print_next_states()
        print()
        self.print_reward()

    def update_action_value(self):
        new_aciton_values = self.reward + self.gamma * self.state_values[self.next_states]

        del self.action_values
        self.action_values = new_aciton_values

    def update_policy(self):
        self.policy.update_policy(self.action_values, self.eps_greedy)

    def update_state_value(self):
        new_state_value = np.sum(self.policy.policy * self.action_values, axis=1)
        
        del self.state_values
        self.state_values = new_state_value

    def learning(self):
        i = 0
        while True:
            i += 1
            v_k = deepcopy(self.state_values)
            
            self.update_action_value()
            self.update_policy()
            self.update_state_value()
            iter_var = np.linalg.norm(self.state_values - v_k)
            print(i, iter_var)
            print(self.state_values)
            
            if iter_var < self.eps_stop:
                break
        print("Total iterations:", i)
        
        

if __name__ == "__main__":
    ex = "..\\resources\\mazes\\2.txt"
    maze = Maze(ex)
    
    vi = ValueIter(maze)
    vi.print()

    print()
    vi.learning()
    print()
    
    vi.print()