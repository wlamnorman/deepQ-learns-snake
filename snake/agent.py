import torch
import random
import numpy as np
from collections import deque
from snake_game_AI import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot


# Optional parameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARING_RATE = 0.001
DISCOUNT_RATE = 0.9  # in [0,1]
N_HIDDEN_LAYER_UNITS: int = 256


class Agent:
    def __init__(self):
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, N_HIDDEN_LAYER_UNITS, 3)
        self.trainer = QTrainer(
            self.model, learning_rate=LEARING_RATE, discount_rate=DISCOUNT_RATE
        )

    def get_state(self, game):
        head = game.snake[0]

        # Clock-wise directions and angles
        cw_dirs = [
            Direction.RIGHT == game.direction,
            Direction.DOWN == game.direction,
            Direction.LEFT == game.direction,
            Direction.UP == game.direction,
        ]
        cw_angles = np.array([0, np.pi / 2, np.pi, -np.pi / 2])

        # pos 0 - check in-front,
        # pos = 1 - check right,
        # pos - -1 check left
        getPoint = lambda pos: Point(
            head.x + BLOCK_SIZE * np.cos(cw_angles[(cw_dirs.index(True) + pos) % 4]),
            head.y + BLOCK_SIZE * np.sin(cw_angles[(cw_dirs.index(True) + pos) % 4]),
        )

        state = [
            game.is_collision(getPoint(0)),  # In which direction is danger
            game.is_collision(getPoint(1)),
            game.is_collision(getPoint(-1)),
            cw_dirs[2],  # The current moving direction
            cw_dirs[0],
            cw_dirs[3],
            cw_dirs[1],
            game.food.x < head.x,  # Is food up/down, left/right
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, new_state, game_over):
        """
        Appends information to memory to allow for re-training with long memory.
        Because memory is a deque it means that if self.memory.__len__() >= MAX_MEMORY
        then the first element into the deque is removed from the left and new element
        added to the right in a very efficient way.
        """
        self.memory.append((state, action, reward, new_state, game_over))

    def train_long_memory(self):
        """
        This allows the AI to train on a number of previous actions determined by BATCH_SIZE.
        """
        if self.memory.__len__() < BATCH_SIZE:
            mini_sample = self.memory
        else:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, new_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, new_states, game_overs)

    def train_short_memory(self, state, action, reward, new_state, game_over):
        """
        This allows the AI to train after each action.
        """
        self.trainer.train_step(state, action, reward, new_state, game_over)

    def get_action(self, state):
        """
        Determines the next action from the current state. Balances exploration
        and explotation.
        """
        # random moves: tradeoff exploration / exploitation
        epsilon = 50 - self.n_games
        action = [0, 0, 0]
        if epsilon > random.randint(0, 100):
            action_randomized = random.randint(0, 2)
            action[action_randomized] = 1
        else:
            state_input = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_input)
            action_predicted = torch.argmax(prediction).item()
            action[action_predicted] = 1
        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_current = agent.get_state(game)
        action = agent.get_action(state_current)
        reward, game_over, score = game.play_step(action)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_current, action, reward, state_new, game_over)
        agent.remember(state_current, action, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
