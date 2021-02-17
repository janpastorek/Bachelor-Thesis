from math import sqrt, pi

import numpy as np
from qiskit.circuit.library import IGate

import CHSH
from CHSH import Game, get_scaler, Agent


class Environment(CHSH.abstractEnvironment):
    """ Creates CHSH environments for quantum strategies, discretizes and states and uses discrete actions """
    def __init__(self, n_questions, evaluation_tactic, max_gates, num_players=2):
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = []
        self.max_gates = max_gates
        self.evaluation_tactic = evaluation_tactic
        self.initial_state = np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0],
                                      dtype=np.float64)
        self.state = self.initial_state.copy()
        self.num_players = num_players
        self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state], dtype=np.float64)
        self.accuracy = self.calc_accuracy([self.measure_analytic() for _ in range(self.n_questions)])
        self.max_acc = self.accuracy
        # input, generate "questions" in equal number
        self.a = []
        self.b = []
        for x in range(2):
            for y in range(2):
                self.a.append(x)
                self.b.append(y)

        self.velocity = 1

    @CHSH.override
    def reset(self):
        self.velocity = 1
        return super().reset()

    def calculate_state(self, history_actions):
        """ Calculates the state according to previous actions"""
        result = []
        self.velocity = 1

        for g in range(self.n_questions):
            # Alice - a and Bob - b share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input, Bob too - eg. a0 if alice gets 0 as input

            self.state = self.initial_state.copy()

            for action in history_actions:

                if action == "biggerAngle":
                    self.velocity *= 2
                    continue
                elif action == "smallerAngle":
                    self.velocity /= 2
                    continue

                gate = self.get_gate(action)
                if gate == IGate:
                    continue

                to_whom = action[0:2]
                gate_angle = np.array([action[4:]], dtype=np.float64) * self.velocity

                if self.a[g] == 0 and to_whom == 'a0':
                    self.state = np.matmul(np.kron(gate((gate_angle * pi / 180).item()).to_matrix(), np.identity(2)),
                                           self.state)

                if self.a[g] == 1 and to_whom == 'a1':
                    self.state = np.matmul(np.kron(gate((gate_angle * pi / 180).item()).to_matrix(), np.identity(2)),
                                           self.state)

                if self.b[g] == 0 and to_whom == 'b0':
                    self.state = np.matmul(np.kron(np.identity(2), gate((gate_angle * pi / 180).item()).to_matrix()),
                                           self.state)

                if self.b[g] == 1 and to_whom == 'b1':
                    self.state = np.matmul(np.kron(np.identity(2), gate((gate_angle * pi / 180).item()).to_matrix()),
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
        result = self.calculate_state(self.history_actions)

        # accuracy of winning CHSH game
        before = self.accuracy
        self.accuracy = self.calc_accuracy(result)

        # reward is the increase in accuracy
        rozdiel_acc = self.accuracy - before
        reward = rozdiel_acc * 100

        # ends when it has applied max number of gates / xxr0
        if self.accuracy >= self.max_acc:
            done = True
            self.max_acc = self.accuracy
            reward += 5 * (1 / (self.count_gates() + 1))

        if self.counter == self.max_gates or self.history_actions[-1] == 'xxr0':
            done = True
            if np.round(self.max_acc, 2) == np.round(self.accuracy, 2):
                reward += 50 * (1 / (self.count_gates() + 1))

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
    ACTIONS2 = ['r' + axis + str(180 / 16 * i) for i in range(1, 3) for axis in 'xyz']
    ACTIONS = ['r' + axis + str(- 180 / 16 * i) for i in range(1, 3) for axis in 'xyz']
    ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in
                            ACTIONS2]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append("xxr0")
    ALL_POSSIBLE_ACTIONS.append("smallerAngle")
    ALL_POSSIBLE_ACTIONS.append("biggerAngle")

    N = 6000
    n_questions = 4
    evaluation_tactic = [[1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0]]
    max_gates = 10
    round_to = 2
    env = Environment(n_questions, evaluation_tactic, max_gates)

    # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
    agent = Agent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=1, eps=1, eps_min=0.01,
                  eps_decay=0.9995, alpha=0.1, momentum=0.9, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS)
    scaler = get_scaler(env, N, ALL_POSSIBLE_ACTIONS, round_to=round_to)
    batch_size = 128

    # store the final value of the portfolio (end of episode)
    game = Game(scaler, round_to=round_to)
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
    np.save(f'.training/train.npy', portfolio_value)

    portfolio_value = game.evaluate_test(agent, env)
    print(portfolio_value)
    a = np.load(f'.training/train.npy')
    print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")
