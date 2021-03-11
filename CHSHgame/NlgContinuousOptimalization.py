from math import sqrt

import numpy as np

import NonLocalGame
from NonLocalGame import BasicAgent, Game
from NlgGeneticOptimalization import CHSHgeneticOptimizer
from agents.DQNAgent import DQNAgent
from models.LinearModel import LinearModel


class Environment(NonLocalGame.abstractEnvironment):

    def __init__(self, n_questions, game_type, max_gates, num_players=2,
                 initial_state=np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0],
                                        dtype=np.float64)):
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = []
        self.max_gates = max_gates
        self.game_type = game_type
        self.initial_state = initial_state
        self.state = self.initial_state.copy()
        self.num_players = num_players
        self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state], dtype=np.float64)
        self.accuracy = self.calc_accuracy([self.measure_analytic() for _ in range(n_questions)])
        self.max_acc = self.accuracy
        self.min_gates = max_gates

        self.optimizer = CHSHgeneticOptimizer(population_size=15, n_crossover=len(self.history_actions) - 1,
                                              mutation_prob=0.10, state=self.initial_state.copy(),
                                              history_actions=self.history_actions.copy(),
                                              game_type=self.game_type,
                                              num_players=self.num_players)
        self.visited = dict()

    @NonLocalGame.override
    def reset(self):
        return super().reset()

    def calculate_new_state(self, action):
        self.history_actions.append(action)
        try:
            actions, accuracy, self.repr_state = self.visited[tuple(self.history_actions)]
        except KeyError:
            self.optimizer.reset(self.history_actions.copy(), len(self.history_actions) - 1)
            actions, accuracy, self.repr_state = self.optimizer.solve(22)
            self.visited[tuple(self.history_actions.copy())] = actions, accuracy, self.repr_state
        return accuracy

    @NonLocalGame.override
    def step(self, action):

        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        # accuracy of winning CHSH game
        # reward is the increase in accuracy
        accuracy_before = self.accuracy
        self.accuracy = self.calculate_new_state(action)
        difference_accuracy = self.accuracy - accuracy_before
        reward = self.reward_qubic(difference_accuracy) - 3

        # print("acc: ", end="")
        # print(self.accuracy)
        #
        # print("rew: ", end="")
        # print(reward)

        if self.counter == self.max_gates or self.history_actions[-1] == 'xxr0':
            done = True

        if done == True:
            print(self.visited[tuple(self.history_actions)][0])
        else:
            self.counter += 1
        return self.repr_state, reward, done


import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    ACTIONS = ['r0']  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in
                            ACTIONS]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append("xxr0")

    N = 4000
    n_questions = 4
    evaluation_tactic = [[1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0]]
    max_gates = 10
    round_to = 3
    env = Environment(n_questions, evaluation_tactic, max_gates)

    # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
    agent = BasicAgent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=1, eps=1, eps_min=0.01,
                       eps_decay=0.9995, alpha=0.01, momentum=0.9, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, model_type=LinearModel)

    hidden_dim = [len(env.repr_state), len(env.repr_state) // 2]
    #
    # agent = DQNAgent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=0.9, eps=1, eps_min=0.01,
    #                  eps_decay=0.9995, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, learning_rate=0.001, hidden_layers=len(hidden_dim),
    #                  hidden_dim=hidden_dim)

    # scaler = get_scaler(env, N, ALL_POSSIBLE_ACTIONS, round_to=round_to)
    batch_size = 128

    # store the final value of the portfolio (end of episode)
    game = Game(round_to=round_to)
    portfolio_value, rewards = game.evaluate_train(N, agent, env)

    # plot relevant information
    NonLocalGame.show_plot_of(rewards, "reward")

    if agent.model.losses is not None:
        NonLocalGame.show_plot_of(agent.model.losses, "loss")

    NonLocalGame.show_plot_of(portfolio_value, "accuracy", [0.85, 0.75])

    # save portfolio value for each episode
    np.save(f'.training/train.npy', portfolio_value)
    portfolio_value = game.evaluate_test(agent, env)
    print(portfolio_value)
    a = np.load(f'.training/train.npy')
    print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")
