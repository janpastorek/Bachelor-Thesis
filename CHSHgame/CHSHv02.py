import random
import abc
from math import sqrt, pi

import numpy as np
from qiskit.extensions import RYGate
import CHSH
from CHSH import Game, get_scaler, Agent


class Environment(CHSH.abstractEnvironment):

    def __init__(self, n_questions, tactic, max_gates, num_players=2):
        self.pointer = 0  # time
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = []
        self.max_gates = max_gates
        self.tactic = tactic
        self.initial_state = np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0],
                                      dtype=np.longdouble)  ## FIX ME SCALABILITY, TO PARAM
        self.state = self.initial_state.copy()
        self.num_players = num_players
        self.repr_state = np.array([x for n in range(self.num_players ** 2) for x in self.state], dtype=np.longdouble)
        self.accuracy = self.calc_accuracy([self.measure_analytic() for i in range(n_questions)])
        self.max_acc = self.accuracy
        # input, generate "questions" in equal number
        self.a = []
        self.b = []
        for x in range(2):
            for y in range(2):
                self.a.append(x)
                self.b.append(y)

    @CHSH.override
    def reset(self):
        self.counter = 1
        self.history_actions = []
        self.state = self.initial_state.copy()  ########## INITIAL STATE
        self.accuracy = self.calc_accuracy([self.measure_analytic() for i in range(n_questions)])
        self.repr_state = np.array([x for n in range(self.num_players ** 2) for x in self.state], dtype=np.longdouble)
        return self.repr_state

    def calculateState(self, history_actions):
        result = []
        for g in range(self.n_questions):
            # Alice - a and Bob - b share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input, Bob too - eg. a0 if alice gets 0 as input

            self.state = self.initial_state.copy()  ########## INITIAL STATE

            for action in history_actions:
                gate = np.array([action[3:]], dtype=np.longdouble)

                if self.a[g] == 0 and action[0:2] == 'a0':  ## FIX ME SCALABILITY, TO PARAM
                    self.state = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                           self.state)

                if self.a[g] == 1 and action[0:2] == 'a1':  ## FIX ME SCALABILITY, TO PARAM
                    self.state = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                           self.state)

                if self.b[g] == 0 and action[0:2] == 'b0':  ## FIX ME SCALABILITY, TO PARAM
                    self.state = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                           self.state)

                if self.b[g] == 1 and action[0:2] == 'b1':  ## FIX ME SCALABILITY, TO PARAM
                    self.state = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                           self.state)

            self.repr_state[g * self.num_players ** 2:(g + 1) * self.num_players ** 2] = self.state.copy()

            result.append(self.measure_analytic())
        return result

    @CHSH.override
    def step(self, action):

        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        # play game
        self.history_actions.append(action)
        result = self.calculateState(self.history_actions)

        # accuracy of winning CHSH game
        before = self.accuracy
        self.accuracy = self.calc_accuracy(result)

        # reward is the increase in accuracy
        rozdiel_acc = self.accuracy - before
        reward = rozdiel_acc * 100

        # skonci, ak uz ma maximalny pocet bran
        if self.accuracy >= self.max_acc:
            self.max_acc = self.accuracy
            reward += 5 * (1 / (self.countGates() + 1))  # alebo za countGates len(history_actuons)

        if self.counter == self.max_gates:
            done = True
            reward += 50 * (1 / (self.countGates() + 1))
            self.counter = 1

        # print("acc: ", end="")
        # print(self.accuracy)
        #
        # print("rew: ", end="")
        # print(reward)

        if done == False:
            self.counter += 1
        return self.repr_state, reward, done


import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ACTIONS2 = ['r' + str(180 / 16 * i) for i in range(1, 9)]
    ACTIONS = ['r' + str(- 180 / 16 * i) for i in range(1, 9)]
    ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in
                            ACTIONS2]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append("xxr0")

    N = 6000
    n_questions = 4
    tactic = [[1, 0, 0, 1],
              [1, 0, 0, 1],
              [1, 0, 0, 1],
              [0, 1, 1, 0]]
    max_gates = 10

    env = Environment(n_questions, tactic, max_gates)  ## FIX ME SCALABILITY, TO PARAM

    # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
    agent = Agent(len(env.repr_state), len(ALL_POSSIBLE_ACTIONS), 0.9, 1, 0.01, 0.9995, 0.001, 0.9,
                  ALL_POSSIBLE_ACTIONS)
    scaler = get_scaler(env, N, ALL_POSSIBLE_ACTIONS)
    batch_size = 128

    # store the final value of the portfolio (end of episode)
    game = Game(scaler)
    portfolio_value, rewards = game.evaluate_train(N, agent, env)

    # plot relevant information
    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.xlabel('Epochs')
    plt.ylabel('Reward')

    plt.plot(rewards)
    plt.show()

    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.plot(agent.model.losses)
    plt.show()

    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.axhline(y=0.853, color='r', linestyle='-')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Win rate')
    plt.plot(portfolio_value)
    plt.show()
    # save portfolio value for each episode
    np.save(f'train.npy', portfolio_value)

    portfolio_value = game.evaluate_test(agent, n_questions, tactic, max_gates, env)
    print(portfolio_value)
    a = np.load(f'train.npy')
    print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")
