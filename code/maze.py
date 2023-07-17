import pandas as pd
import model_config as mc

'''
Whats maze?
Given a maze, we know the model, and we need to solve the maze and give below outputs
1. policy - for every block in our maze 
2. state value / action value


Model
1. states set
* accessible
* block
* target

2. actions set
* up
* down
* left
* right
* stay
'''

class ErrorMazeException(Exception):
    pass

class Maze:
    def __init__(self, path):
        maze = open(path).read().strip()
        if not maze:
            raise ErrorMazeException

        self.maze = self.__parse_maze__(maze)
        self._states_set = list(set([j for i in self.maze for j in i]))
        self._actions_set = ["up", "down", "right", "left", "stay"]
        
        self.rows = len(self.maze)
        self.cols = len(self.maze[0])
        self.states_n = len(self._states_set)
        self.actions_n = len(self._actions_set)

    def __len__(self):
        return self.rows * self.cols
    
    def __parse_maze__(self, maze):
        return [[ch for ch in line.strip().split(" ")] for line in maze.split("\n")]

    def print(self):
        print("Maze:")
        print(pd.DataFrame(self.maze))

    def get_next_state(self, state, action):
        # state is a real number starts from 0
        row = state // self.cols
        col = state % self.cols

        if (row == 0 and action == "up") \
            or (row == self.rows - 1 and action == "down") \
            or (col == 0 and action == "left") \
            or (col == self.cols - 1 and action == "right"):
            return state
        
        next_row = row
        next_col = col
        if action == "up":
            next_row -= 1
        elif action == "down":
            next_row += 1
        elif action == "left":
            next_col -= 1
        elif action == "right":
            next_col += 1

        # # 不允许进入block
        # if self.maze[next_row][next_col] == "b":
        #     return state

        return next_row * self.cols + next_col

    def get_reward(self, state, action):
        row = state // self.cols
        col = state % self.cols
        state_ch = self.maze[row][col]

        if (row == 0 and action == "up") \
            or (row == self.rows - 1 and action == "down") \
            or (col == 0 and action == "left") \
            or (col == self.cols - 1 and action == "right"):
            return mc.r_b

        next_state = self.get_next_state(state, action)
        next_row = next_state // self.cols
        next_col = next_state % self.cols
        next_state_ch = self.maze[next_row][next_col]

        if next_state_ch == "a":
            return mc.r_a
        elif next_state_ch == "b":
            return mc.r_b
        elif next_state_ch == "t":
            return mc.r_t
        
        return mc.r_unknown

    def get_gamma(self):
        return mc.gamma

    def gen_exp(self, curr_state, curr_action, behavior_policy):
        reward = self.get_reward(curr_state, curr_action)
        # s_{t+1}
        next_state = self.get_next_state(curr_state, curr_action)
        # a_{t+1}
        next_action = self._actions_set[behavior_policy.choose_action(next_state)]

        return (curr_state, curr_action, reward, next_state, next_action)


if __name__ == "__main__":
    ex = "..\\resources\\mazes\\0.txt"
    maze = Maze(ex)
    maze.print()
    print()

    [print(action, maze.get_next_state(0, action)) for action in maze._actions_set]