import random
from math import sqrt, pi

import numpy as np
from qiskit.extensions import RYGate
from CHSH import Game, Agent, get_scaler
import CHSH


class Environment(CHSH.abstractEnvironment):

    def __init__(self, n_questions, tactic, max_gates):
        self.pointer = 0  # time
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = ['a0r0' for i in range(max_gates)]
        self.max_gates = max_gates
        self.tactic = tactic
        self.initial_state = np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.longdouble)
        self.state = self.initial_state.copy()
        self.accuracy = 0.25
        self.num_players = 2
        self.repr_state = [0 for i in range(max_gates)]

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
        self.history_actions = ['a0r0' for i in range(self.max_gates)]
        self.accuracy = 0.25
        self.state = self.initial_state.copy()  ########## INITIAL STATE
        self.repr_state = [0 for i in range(self.max_gates)]
        return self.repr_state

    @CHSH.override
    def step(self, action):

        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        # play game
        result = []
        self.repr_state[self.counter - 1] = action[1]
        self.history_actions[self.counter - 1] = action[0]

        for g in range(self.n_questions):
            # Alice - a and Bob - b share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input, Bob too - eg. a0 if alice gets 0 as input

            self.state = self.initial_state.copy()  ########## INITIAL STATE

            for action in self.history_actions:
                gate = np.array([action[3:]], dtype=np.longdouble)

                if self.a[g] == 0 and action[0:2] == 'a0':
                    self.state[:4] = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                               self.state[:4])

                if self.a[g] == 1 and action[0:2] == 'a1':
                    self.state[:4] = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                               self.state[:4])

                if self.b[g] == 0 and action[0:2] == 'b0':
                    self.state[:4] = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                               self.state[:4])

                if self.b[g] == 1 and action[0:2] == 'b1':
                    self.state[:4] = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                               self.state[:4])

            result.append(self.measure_analytic())

        # for i in result:
        #     print(i)

        # accuracy of winning CHSH game
        before = self.accuracy
        self.accuracy = self.calc_accuracy(self.tactic, result)

        # reward is the increase in accuracy
        rozdiel_acc = self.accuracy - before
        reward = rozdiel_acc * 100

        # skonci, ak uz ma maximalny pocet bran alebo presiahol pozadovanu uroven self.accuracy
        if self.accuracy >= 0.83:
            done = True
            reward += 500 * (1 / (len(self.history_actions) + 1))

        elif self.counter == self.max_gates:
            done = True
            # reward = -100

        print("acc: ", end="")
        print(self.accuracy)

        print("rew: ", end="")
        print(reward)

        if done == False:
            self.counter += 1
        return sorted(self.repr_state), reward, done


import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ACTIONS2 = ['r' + str(180 / 16 * i) for i in range(0, 9)]
    ACTIONS = ['r' + str(- 180 / 16 * i) for i in range(1, 9)]
    ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in
                            ACTIONS2]  # place one gate at some place

    N = 6000
    n_questions = 4
    tactic = [[1, 0, 0, 1],
              [1, 0, 0, 1],
              [1, 0, 0, 1],
              [0, 1, 1, 0]]
    max_gates = 10

    env = Environment(n_questions, tactic, max_gates)

    # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
    agent = Agent(len(env.repr_state), len(ALL_POSSIBLE_ACTIONS), 0.9, 1, 0, 0.995, 0.001, 0.9, ALL_POSSIBLE_ACTIONS)
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

    plt.axhline(y=0.853, color='r', linestyle='-')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Win rate')
    plt.plot(portfolio_value)
    plt.show()
    # save portfolio value for each episode
    np.save(f'train.npy', portfolio_value)

    portfolio_value = game.evaluate_test(agent, n_questions, tactic, max_gates)
    print(portfolio_value)

    a = np.load(f'train.npy')

    print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

    plt.plot(a)
    plt.show()
