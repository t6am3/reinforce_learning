from value_iteration import ValueIter
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from utils import gen_random_prob
from maze import Maze
import model_config as mc
from policy import Policy


'''
sarsa is a temporal differential on-policy
q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - alpha_t(s_t, a_t) * (q_t(s_t, a_t) - [r_{t+1} + gamma * q_t(s_{t+1}, a_{t+1}})])
sarsa is a on-policy, so behavior policy will update
'''
class Sarsa(ValueIter):
    def learning(self, start_state=0, start_action_idx=0, exps_n=1, max_length=10000):
        start_action = self.maze._actions_set[start_action_idx]

        for i in range(exps_n):
            curr_state = start_state
            curr_action = start_action
            
            j = 1
            while j < max_length and self.maze.maze[curr_state // self.maze.cols][curr_state % self.maze.cols] != "t":
                j += 1
                visit = self.maze.gen_exp(curr_state, curr_action, self.policy)
                # print("enter visit: ", visit)
                curr_state, curr_action, reward, next_state, next_action = visit
                curr_action_idx = self.maze._actions_set.index(curr_action)
                next_action_idx = self.maze._actions_set.index(next_action)

                td_target = reward + self.gamma * self.action_values[next_state][next_action_idx]
                td_err = self.action_values[curr_state][curr_action_idx] - td_target
                curr_new_action_value = self.action_values[curr_state][curr_action_idx] - self.alpha * td_err
                
                self.update_action_value(curr_state, curr_action_idx, curr_new_action_value)
                self.update_state_value()
                self.update_policy()

                curr_state = next_state
                curr_action = next_action

                # self.print_policy()
                # self.print_action_values()

            print(f"gen exp end. {j} visit generated")
    
    def update_action_value(self, state, action_idx, new_value):
        self.action_values[state, action_idx] = new_value

if __name__ == "__main__":
    ex = "..\\resources\\mazes\\2.txt"
    maze = Maze(ex)
    
    sarsa = Sarsa(maze, eps_greedy=0.1)
    sarsa.print()

    print()
    sarsa.learning(exps_n=100)

    print()
    sarsa.print()