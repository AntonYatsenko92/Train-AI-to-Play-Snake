# Training AI to Play Snake

## Project Goal

The purpose of this project is to develop a Q-Learning model that can teach an AI agent how to play snake. In doing so, the project aims to explore the basics of Reinforcement Learning (RL) and how it can be used to train AI to play snake and then optimized based on the hyperparameters available to maximize learning efficiency (defined by the speed in which the agent learns to play the snake game). 

## Reinforcement/Q-Learning Basics

The project was divided into four (4) main steps. The steps are summarized below and will be detailed out throughout this README file:

**Step 1:** Create base snake game and training agent code using Python

**Step 2:** Identify base reinforcement learning model used that will be used to train the agent

**Step 3:** Train model using agent and optimize available hyperparameters 

**Step 4:** Analyze data based on hyperparameter optimization and train final based on hyperparameters

Before beginning with the coding, a quick refresher on Reinforcement Learning (RL), and more specifically Q-Learning, is necessary. RL is the process of training a computer, typically referred to as an agent, to make optimal decisions within a given environment based on positive and negative rewards it receives. For example, in this model, the agent was rewarded with points every time it correctly ate a piece of food and lost points every time the game was over. This constant iterative feedback loop is the basis of RL, teaching an agent how to successfully learn to play a game within an environment.

Q-learning is a subset of RL which referes to Q-value derived from the Bellman Equation, which represents the value of taking an action (a) in a state (s). Let's introduce the Bellman Equation first and discuss in more detail:

\[ V(s) = \max_{a} \left( R(s, a) + \gamma \sum_{s'} P(s' | s, a) V(s') \right) \]

Let's break this down into its components:

**Q(s,a):** Commonly referred to as the Q-Value of a decision based on a state (s) and action (a). In the game of snake, each "tile" refers to a state where the agent can choose a direction to go (action). Depending on the action the agent chooses at each state, a reward may or may not be given.

The Q-Value is constantly updated as the agent continues through iterations of the game to learn which actions at which states result in positive and negative rewards with the ultimate goal of maximizing total reward.

**&alpha;:** Alpha, referred to as the learning rate, provides a scalar multiplier to how much emphasis the agent places on future and current state rewards. This is represented by the remainder of the equation, where the agent slowly builds up knowledge and learns its surroundings. Typically, α is a hyperparameter that can be set by the programmer. A learning rate that is too high risks convergence as the agent will continuously hunt for an optimal process, whereas a learning rate that is too small risks either converging too slowly or getting stuck at a local minimum.

**r:** The reward is another hyperparameter the programmer can specify to provide positive and negative feedback to the agent based on the choices it makes. For the snake game, the base model had the following reward structure:

1) +10 points for eating a piece of food

2) -10 points for dying

3) 0 points otherwise

**&gamma;:** Gamma, referred to as the discount factor, provides a discount rate to future rewards. This incentivizes the agent to balance exploration and exploitation. A high gamma encourages the agent to prioritize immediate rewards, potentially repeating past actions to maximize reward by looking at future states. Conversely, a low gamma results in the agent consistently exploring without effectively learning how to exploit its environment.


### Step 1: Create Base Snake Game & Agent Code

The first step in this process was to create the base snake game. This was accomplished with support from Patrick Lobner's Reinforcement Learning video series, resulting in the development of the base snake game using the Pygame package.

The game was designed to be controlled similarly to a traditional snake game, with four inputs to control the snake in the up, down, left, and right directions. Once the base game was created and tested, the script was modified to allow control by the agent. This was encapsulated in the following class, which initializes, quits once complete, and automatically restarts. A snippet of the code is shown below, with the full code available in the "game.py" file.


```

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
    
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

```

With the game created, the next step was to develop the agent script, responsible for continuously running the game and training the AI to play snake. The agent model was implemented using PyTorch, a common Reinforcement Learning package in Python. It imports functionalities from the game, model, and helper scripts to execute the training process.

The agent operates by randomly selecting an action based on its current state and evaluating all possible moves. As it continues to train and learn, associating positive rewards with obtaining food and negative rewards with crashing into obstacles and ending the game, the agent's performance improves.

Overall, the function takes the following inputs: self, state, action, reward, next_state, and done. These inputs assist the agent in making decisions about its next action, with 'done' indicating whether the game has ended and needs to be reset.

```
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
```

Once the game is over, the agent code was enhanced to plot the mean score and the previous score on a graph for visualization purposes. Additionally, both the graph and scores were configured to be exported every 200 rounds, as training was frequently conducted passively to ensure that data was retained for analysis in Step #3. Below are some code excerpts, with the complete code available in the "agent.py" file.
```
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
             if agent.n_games % 200 == 0:
                filename = f'plot_{agent.n_games}.png'
                plot_scores_mean(plot_scores, plot_mean_scores, prev_score, filename=filename)

            # Update previous score
            prev_score = score    
```
           if agent.n_games % 200 == 0:
                filename = f'plot_{agent.n_games}.png'
                plot_scores_mean(plot_scores, plot_mean_scores, prev_score, filename=filename)

            # Update previous score
            prev_score = score
```
 
### Step 2: Create Reinforcement Learning Model

The chosen reinforcement learning model was a simple linear model consisting of an input layer, hidden layer, and output layer, implemented using PyTorch. The model was intended to be invoked by the agent, with its primary objective being to predict future Q-values based on the input parameters and hyperparameters specified in the training model. It utilizes the same inputs outlined above: state, action, reward, next_state, and done. Below is a code excerpt from the "model.py" file illustrating this.

```
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
```

As part of the model optimization process, more complex models were developed; however, they failed to converge. While this issue has not been extensively investigated, one hypothesis suggests that the relative simplicity of training the agent in this environment might be a contributing factor. It's possible that the failure of more complex models to converge is due to a small learning rate or continuous overshooting.

### Step 3: Optimize Model Hyperparameters

With the models created, the next step was to optimize all available hyperparameters to maximize the learning rate of the game once the base model was established. The base model had the following hyperparameters:

* **NN Hidden Layer:**: 256
* **Learning Rate:** 0.001
* **Gamma:** 0.9
* **Epsilon:** 100

For equal comparison, a threshold of 1000 games was chosen with the goal of evaluating the average score at each 200-game interval and the final average score for each model created. The results of the base model are shown below:

| 200 Games | 400 Games | 600 Games | 800 Games | 1000 Games |
|----------|----------|----------|----------|----------|
| 11.54 | 22.66 | 25.90 | 27.27 | 27.99 |

Next, lets discuss each hyperparameter in detail and what can happen as values increase from small to large values. The results of the hyperparameter optimization will be summarized in Step #4 below. 

#####Neural Network Hidden Layers

The number of hidden layers in a neural network represents the number of different pathways the model can utilize to establish a relationship between the input and output parameters. This concept mirrors the functionality of neurons in the brain. It's important to note that from a modeling perspective, the number of layers signifies the maximum number of hidden layers the model has access to, but it doesn't necessarily mean the model will utilize all of them. While increasing the number of layers can enhance model complexity, it may also lead to overfitting.

**Values Tested:** 250, 500, 750, 1000, 10000

####Learning Rate

The learning rate represents the magnitude of the 'step' a model takes when updating its parameters during training. If the learning rate is too high, the model may oscillate around the optimal solution or fail to converge, as it overshoots the optimal parameters. Conversely, if the learning rate is too small, the model may converge too slowly or get stuck in a local minimum of the loss function it is attempting to minimize.

**Values Tested:** 0.1, 0.01, 0.001, 0.0001, 0.00001

####Gamma (&gamma;)

The discount factor (γ) represents the weight or importance the agent places on future rewards. By learning the environment through gameplay, the agent can anticipate future rewards, represented by the estimated Q-value (Q'(s',a')), such as obtaining the next food item. The purpose of the discount factor is to strike a balance between exploration (learning about the environment) and exploitation (maximizing reward score by obtaining food).

A γ value that is too low will lead to limited exploration by the agent, as it prioritizes immediate rewards without adequately learning about the environment. Conversely, a γ value that is too high can result in excessive exploration, which may reduce the average score achieved by the agent.

**Values Tested:** 0.1 - 0.9

####Epsilon (&epsilon;)

The exploration-exploitation trade-off in reinforcement learning is further influenced by the epsilon (ε) parameter. Epsilon represents the probability with which the agent chooses to explore new actions rather than exploiting the current best action based on its learned policy. If epsilon is too low, the agent will primarily exploit its current knowledge, potentially missing out on discovering better strategies or exploring uncharted territories of the environment. Conversely, if epsilon is too high, the agent may overly prioritize exploration, neglecting the exploitation of known effective strategies and thereby hindering its ability to achieve high rewards.

**Values Tested:** 0.1 - 0.9

####Reward Value

The final values tested was value of positive or negative reward built into the model to see how it would affect model performance. Positive and negative reward balances were altered to see affect on model performance.

**Values Tested:** +/- 10 - 20


### Step 4: Analyze Results and Create Final Model 

#####Neural Network Hidden Layers Results

|  | 200 Games | 400 Games | 600 Games | 800 Games | 1000 Games |
|----------|----------|----------|----------|----------|----------|
| NN = 250 | 18.59 | 25.55 | 27.04 | 28.12 | 28.95 |
| NN = 500 | 19.39 | 26.63 | 28.71 | 29.28 | 29.65 |
| NN = 750 | 19.57 | 25.95 | 27.57 | 28.39 | 28.71 |
| NN = 1000 | 21.59 | 27.06 | 28.30 | 29.23 | 30.12 |
| NN = 10000 | 19.70 | 25.52 | 28.05 | 29.32 | 29.62 |

Based on the results we can see that NN = 1000 performed best and will be used in the final model. 

#####Learning Rate


|  | 200 Games | 400 Games | 600 Games | 800 Games | 1000 Games |
|----------|----------|----------|----------|----------|----------|
| LR = 0.1 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 |
| LR = 0.01 | 0.21 | 0.11 | 0.08 | 0.06 | 0.05 |
| LR = 0.001 | 20.23 | 26.61 | 28.30 | 29.77 | 30.36 |
| LR = 0.0001 | 0.94 | 14.53 | 20.60 | 23.96 | 25.56 |
| LR = 0.00001 | 1.42 | 2.12 | 1.58 | 2.57 | 4.48 |

Based on the results LR = 0.01 performed best and will be used in the final model. 

####Gamma (&gamma;)

|  | 200 Games | 400 Games | 600 Games | 800 Games | 1000 Games |
|----------|----------|----------|----------|----------|----------|
| Gamma = 0.1 | 2.27 | 8.69 | 8.15 | 8.15 | 6.56 |
| Gamma = 0.2 | 6.96 | 18.60 | 23.14 | 22.42 | 21.00 |
| Gamma = 0.3 | 18.81 | 19.63 | 23.71 | 26.65 | 27.87 |
| Gamma = 0.4 | 18.39 | 27.95 | 30.40 | 30.67 | 31.13 |
| Gamma = 0.5 | 21.17 | 27.87 | 29.74 | 30.97 | 31.75 |
| Gamma = 0.6 | 22.50 | 28.24 | 28.06 | 29.91 | 31.02 |
| Gamma = 0.7 | 24.36 | 31.35 | 33.60 | 34.98 | 35.31 |
| Gamma = 0.8 | 22.11 | 26.73 | 29.02 | 29.49 | 29.97 |
| Gamma = 0.9 | 19.25 | 25.47 | 27.39 | 28.66 | 29.31 |

Based on the results Gamma = 0.7 performed best and will be used in the final model. 

####Epsilon (&epsilon;)

|  | 200 Games | 400 Games | 600 Games | 800 Games | 1000 Games |
|----------|----------|----------|----------|----------|----------|
| Epsilon = 100 | 19.17 | 27.67 | 31.50 | 33.87 | 35.31 |
| Epsilon = 200 | 2.40 | 18.95 | 24.93 | 27.62 | 29.79 |
| Epsilon = 300 | 0.10 | 10.70 | 18.08 | 22.32 | 23.77 |
| Epsilon = 400 | 0.08 | 0.95 | 13.80 | 19.80 | 22.76 |
| Epsilon = 500 | 0.10 | 0.18 | 7.50 | 13.44 | 17.84 |

Based on the results Epsilon = 100 performed best and will be used in the final model. 

####Reward Value

|  | 200 Games | 400 Games | 600 Games | 800 Games | 1000 Games |
|----------|----------|----------|----------|----------|----------|
| Reward = +10/-10 | 18.77 | 27.41 | 30.52 | 31.54 | 32.36 |
| Reward = +20/-10 | 20.30 | 28.39 | 31.04 | 31.82 | 32.35 |
| Reward = +30/-10 | 18.77 | 28.54 | 31.88 | 33.12 | 33.60 |
| Reward = +40/-10 | 17.86 | 24.84 | 28.88 | 30.21 | 30.84 |
| Reward = +50/-10 | 16.01 | 27.36 | 30.37 | 32.26 | 33.44 |
| Reward = +15/-5 | 19.68 | 28.76 | 32.11 | 34.77 | 36.48 |
| Reward = +20/-5 | 19.11 | 27.07 | 28.86 | 30.15 | 30.77 |

Based on the results Reward = +15/-5 performed best and will be used in the final model. 

####Base vs. Final Model Comparison

|  | 200 Games | 400 Games | 600 Games | 800 Games | 1000 Games |
|----------|----------|----------|----------|----------|----------|
| Base Model | 18.59 | 25.55 | 27.04 | 28.12 | 28.95 |
| Final Model | 19.68 | 28.76 | 32.11 | 34.77 | 36.48 |

Based on the results we can see a significant improvement occuring after the 200 game mark with convergence of both models at their peak average score at 1000 games. The final model parameters are summarized below:

**Neural Network Hidden Layers:** 1000

**Learning Rate:** 0.001

**Gamma (&gamma;):** 0.7

**Epsilon (&epsilon;):** 100

**Reward Value:** +15 / -5 

## Challenges & Next Steps

####Challenges

**Challenge #1: Game Size Limited Model Growth** The game had an upper limit based on screen resolution before the snake ran out of room to grow. It's uncertain whether the same parameters would have been optimal on a larger resolution game format.


**Challenge #2: Snake Was Unaware of Tail** Game controls were simplified by only focusing on the snake's direction, resulting in the snake being unaware of the position of its tail. Consequently, the snake could trap itself inside its body, as it couldn't factor this into its decision algorithm.

**Challenge #3: Model Abiguity** Three models were chosen, with the two more complicated ones unable to converge. Troubleshooting models was challenging due to the "black box" parameters of most Q-Learning algorithms.

####Next Steps

**Next Steps #1: Optimize Models Based on Different Game Sizes** Re-validate optimal hyperparameters based on different game resolution sizes to understand whether performance changes as the game size increases.

**Next Steps #2: Modify Programming for Tail Awareness** Modify game programming to enhance the snake's awareness of its own tail position. This aims to determine if this alteration will influence the agent's behavior to avoid getting stuck inside its own tail in later game stages.

**Next Steps #3: Test Model Abiguity** Continue exploring new Q-Learning models or additional Reinforcement Learning models to gain a deeper understanding of why more complicated models did not converge.

