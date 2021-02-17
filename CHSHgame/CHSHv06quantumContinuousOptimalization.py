from math import sqrt

import numpy as np

import CHSH
from CHSH import get_scaler, Agent, Game
from CHSHv05quantumGeneticOptimalization import CHSHgeneticOptimizer


class Environment(CHSH.abstractEnvironment):

    def __init__(self, n_questions, evaluation_tactic, max_gates, num_players=2):
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = []
        self.max_gates = max_gates
        self.evaluation_tactic = evaluation_tactic
        self.initial_state = np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0],
                                      dtype=np.longdouble)
        self.state = self.initial_state.copy()
        self.num_players = num_players
        self.repr_state = np.array([x for n in range(self.num_players ** 2) for x in self.state], dtype=np.longdouble)
        self.accuracy = self.calc_accuracy([self.measure_analytic() for i in range(n_questions)])
        self.max_acc = self.accuracy
        self.min_gates = max_gates

        self.optimizer = CHSHgeneticOptimizer(population_size=15, n_crossover=len(self.history_actions) - 1,
                                              mutation_prob=0.10, state=self.initial_state,
                                              history_actions=self.history_actions,
                                              evaluation_tactic=self.evaluation_tactic,
                                              num_players=self.num_players)
        self.visited = dict()

    @CHSH.override
    def reset(self):
        return super().reset()

    def calculateNewStateAccuracy(self, action):
        self.history_actions.append(action)
        try:
            actions, accuracy, self.repr_state = self.visited[tuple(self.history_actions)]
        except KeyError:
            self.optimizer.reset(self.history_actions, len(self.history_actions) - 1)
            actions, accuracy, self.repr_state = self.optimizer.solve(22)
            self.visited[tuple(self.history_actions)] = actions, accuracy, self.repr_state
        return accuracy

    @CHSH.override
    def step(self, action):

        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        # accuracy of winning CHSH game
        # reward is the increase in accuracy
        accuracyBefore = self.accuracy
        self.accuracy = self.calculateNewStateAccuracy(action)
        reward, done = self.rewardOnlyBest(accuracyBefore, done)

        # print("acc: ", end="")
        # print(self.accuracy)
        #
        # print("rew: ", end="")
        # print(reward)

        if done == True:
            print(self.visited[tuple(self.history_actions)][0])
        else:
            self.counter += 1
        return self.repr_state, reward, done

    def rewardOnlyBest(self, accuracyBefore, done):
        reward = self.accuracy - accuracyBefore
        reward *= 100

        # always award only the best (who is best changes through evolution)
        if np.round(self.accuracy, 2) > np.round(self.max_acc, 2):
            reward += 50 * (self.max_acc - self.accuracy)
            self.min_gates = len(self.history_actions)
            self.max_acc = self.accuracy
        elif np.round(self.accuracy, 2) == np.round(self.max_acc, 2):
            if self.min_gates > len(self.history_actions):
                self.min_gates = len(self.history_actions)

        # end when it has applied max number of gates / xxr0
        if self.counter == self.max_gates or self.history_actions[-1] == "xxr0":
            done = True
            if np.round(self.max_acc, 2) == np.round(self.accuracy, 2) and self.min_gates == self.count_gates():
                reward = 5000 * (1 / (self.count_gates() + 1)) * self.accuracy
            elif np.round(self.max_acc, 2) == np.round(self.accuracy, 2):
                reward -= 1000 * (self.count_gates() + 1) / self.accuracy
            else:
                reward -= 10000 * (self.count_gates() + 1) / self.accuracy  # alebo tu dam tiez nejaky vzorcek

        return reward, done

    def rewardPositiveDifference(self, accuracyBefore, done):
        reward = self.accuracy - accuracyBefore
        if self.counter == self.max_gates or self.history_actions[-1] == "xxr0":
            done = True
        return reward, done


import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

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
    max_gates = 6
    discretizeByRoundintTo = 2
    env = Environment(n_questions, evaluation_tactic, max_gates)

    # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
    agent = Agent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=1, eps=1, eps_min=0.01,
                  eps_decay=0.9995, alpha=0.5, momentum=0.5, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS)
    scaler = get_scaler(env, N, ALL_POSSIBLE_ACTIONS, round_to=discretizeByRoundintTo)
    batch_size = 128

    # store the final value of the portfolio (end of episode)
    game = Game(scaler, round_to=discretizeByRoundintTo)
    portfolio_value, rewards = game.evaluate_train(N, agent, env)

    # plot relevant information
    # reward
    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.xlabel('Epochs')
    plt.ylabel('Reward')

    plt.plot(rewards)
    plt.show()

    # agent loss function

    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.plot(agent.model.get_losses())
    plt.show()

    # win rate
    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.axhline(y=0.853, color='r', linestyle='-')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Win rate')
    plt.plot(portfolio_value)
    plt.show()

    # save portfolio value for each episode
    np.save(f'.training/train.npy', portfolio_value)
    portfolio_value = game.evaluate_test(agent, env)
    print(portfolio_value)
    a = np.load(f'.training/train.npy')
    print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")
