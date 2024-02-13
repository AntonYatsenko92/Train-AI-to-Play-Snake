import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.7
EPS = 100

GRAPH_NAME = f'Final Model'

class Agent: 

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = GAMMA # discount rate
        self.memory = deque(maxlen = MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 1000, 3) ### Change 256 parameter for neural network size ###
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
        

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
    
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u, 
            dir_d,

            # Food location
            game.food.x < game.head.x, # Food left
            game.food.x > game.head.x, # Food right
            game.food.y < game.head.y, # Food up
            game.food.y > game.head.y # Food down
        ]
    
        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)
 
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration vs. exploitation
        self.epsilon = EPS - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
def save_scores_to_dataframe(dataframe, games, score):
    dataframe = pd.concat([dataframe, pd.DataFrame([{'Game': games, 'Score': score}])], ignore_index=True)
    return dataframe

def export_dataframe_to_csv(dataframe, games):
    if games % 200 == 0 and games > 0:
        filename = f'scores_{games}.csv'
        dataframe.to_csv(filename, index=False)
        print(f'DataFrame exported to {filename}')


def plot_scores_mean(scores, mean_scores, prev_score, filename=None):
    plt.clf()  # Clear the previous plot

    plt.plot(scores, label='Score', color='blue')
    plt.plot(mean_scores, label='Mean Score', color='red')

    plt.xlabel('Games')
    plt.ylabel('Score')

    # Set the title
    plt.title(GRAPH_NAME)

    plt.legend()

    # Display mean score as text annotation
    plt.text(len(scores) - 1, mean_scores[-1], f'Mean: {mean_scores[-1]:.2f}', fontsize=8, color='red')

    if filename:
        plt.savefig(str(filename))  # Convert filename to string
    else:
        plt.show()

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    prev_score = 0  # Initialize previous score
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    df_scores = pd.DataFrame(columns=['Game', 'Score'])  # Initialize DataFrame

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            df_scores = save_scores_to_dataframe(df_scores, agent.n_games, score)

            export_dataframe_to_csv(df_scores, agent.n_games)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot_scores_mean(plot_scores, plot_mean_scores, prev_score)

            if agent.n_games % 200 == 0:
                filename = f'plot_{agent.n_games}.png'
                plot_scores_mean(plot_scores, plot_mean_scores, prev_score, filename=filename)

            # Update previous score
            prev_score = score

if __name__ == '__main__':
    train()