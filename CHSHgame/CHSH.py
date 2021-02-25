from math import sqrt

import matplotlib.pyplot as plt
from qiskit.extensions import RYGate, RZGate, RXGate, IGate
from sklearn.preprocessing import StandardScaler

from agents.BasicAgent import BasicAgent
from agents.DQNAgent import DQNAgent
from models.LinearModel import LinearModel


def get_scaler(env, N, ALL_POSSIBLE_ACTIONS, round_to=2):
    """:returns scikit-learn scaler object to scale the states"""
    # Note: you could also populate the replay buffer here
    states = []
    for _ in range(N):
        action = np.random.choice(ALL_POSSIBLE_ACTIONS)
        state, reward, done = env.step(action)
        states.append(np.round(state, round_to))

        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def show_plot_of(plot_this, label, place_line_at=()):
    # plot relevant information
    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)

    for pl in place_line_at:
        plt.axhline(y=pl, color='r', linestyle='-')

    plt.xlabel('Epochs')
    plt.ylabel(label)

    plt.plot(plot_this)
    plt.show()


def override(f): return f


from abc import ABC, abstractmethod


class abstractEnvironment(ABC):
    """ abstract environment to create CHSH framework """

    @abstractmethod
    def reset(self):
        """Return initial_time_step."""
        self.counter = 1
        self.history_actions = []
        self.state = self.initial_state.copy()
        self.accuracy = self.calc_accuracy([self.measure_analytic() for _ in range(self.n_questions)])
        self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state] + [self.count_gates()], dtype=np.float64)
        return self.repr_state

    @abstractmethod
    def step(self, action):
        """Apply action and return new time_step."""
        pass

    def measure_analytic(self):
        """ :returns probabilities of 00,01,10,11 happening in matrix """
        weights = [abs(a) ** 2 for a in self.state]
        return weights

    def calc_accuracy(self, result):
        """ :returns winning accuracy / win rate based on winning evaluation_tactic """
        win_rate = 0
        for x, riadok in enumerate(self.evaluation_tactic):
            for y, stlpec in enumerate(riadok):
                win_rate += (stlpec * result[x][y])
        win_rate = win_rate * 1 / len(self.evaluation_tactic)
        return win_rate

    def count_gates(self):
        """ :returns count of relevant gates """
        count = 0
        for action in self.history_actions:
            if action in {"xxr0"}:
                pass
            elif action in {"smallerAngle", "biggerAngle"}:
                count += 0.5
            else:
                count += 1

        return count

    def get_gate(self, action):
        """ :returns gate got from string code of action """
        gate = action[2:4]
        if gate == "rx":
            return RXGate
        elif gate == "ry":
            return RYGate
        elif gate == "rz":
            return RZGate
        else:
            return IGate


import random

import warnings

warnings.filterwarnings('ignore')
import pickle
import numpy as np


class Game:
    """ creates CHSH game framework for easier manipulation """

    def __init__(self, scaler=None, round_to=2, batch_size=32):
        self.scaler = scaler
        self.round_to = round_to
        self.batch_size = batch_size

    def play_one_episode(self, agent, env, DO):
        """ Plays one episode of CHSH training
        :returns last accuracy acquired and rewards from whole episode """
        # in this version we will NOT use "exploring starts" method
        # instead we will explore using an epsilon-soft policy
        state = env.reset()
        if self.scaler is not None: state = self.scaler.transform([state])
        else: state = np.array([np.round(state, self.round_to)], dtype=np.float64)
        done = False

        # be aware of the timing
        # each triple is s(t), a(t), r(t)
        # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)

        rew_accum = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action[0])
            if self.scaler is not None: next_state = self.scaler.transform([np.round(next_state, self.round_to)])
            else: next_state = np.array([np.round(next_state, self.round_to)], dtype=np.float64)
            if DO == 'train':
                if type(agent) == BasicAgent:
                    agent.train(state, action[1], reward, next_state, done)
                elif type(agent) == DQNAgent:
                    agent.update_replay_memory(state, action[1], reward, next_state, done)
                    agent.replay(self.batch_size)
            state = next_state.copy()
            rew_accum += reward
        print(env.history_actions)
        return env.accuracy, rew_accum

    def evaluate_train(self, N, agent, env):
        """ Performes the whole training of agent in env in N steps
        :returns portfolio value and rewards for all episodes - serves to plot how it has trained"""
        DO = "train"

        portfolio_value = []
        rewards = []

        for e in range(N):
            val, rew = self.play_one_episode(agent, env, DO)
            print('episode:', end=' ')
            print(e, end=' ')
            print('acc:', end=' ')
            print(val)
            print('rew:', end=' ')
            print(rew)

            portfolio_value.append(val)  # append episode end portfolio value
            rewards.append(rew)

        # save the weights when we are done
        if DO == 'train':
            # # save the DQN
            agent.save(f'.training/linear.npz')

            # save the scaler
            with open(f'.training/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

        return portfolio_value, rewards

    def evaluate_test(self, agent, env):
        """ Tests what has the agent learnt in N=1 steps :returns accuracy and reward """
        DO = "test"

        portfolio_value = []
        if DO == 'test':
            N = 1

            # then load the previous scaler
            if self.scaler != None:
                with open(f'.training/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)

            # make sure epsilon is not 1!
            # no need to run multiple episodes if epsilon = 0, it's deterministic
            agent.epsilon = 0

            # load trained weights
            agent.load(f'.training/linear.npz')

        # play the game num_episodes times

        for e in range(N):
            val = self.play_one_episode(agent, env, DO)
            print('Test value:', end=' ')
            print(val)

            portfolio_value.append(val)  # append episode end portfolio value

        return portfolio_value


import itertools


def generate_only_interesting_tactics(size=4):
    """ Generates only interesting evaluation tactics
    because some are almost duplicates and some will have no difference between classic and quantum strategies. """
    product = list(itertools.product([0, 1], repeat=size))
    tactics = list(itertools.product(product, repeat=size))
    print(len(tactics))
    interesting_evaluation = dict()
    for tactic in tactics:
        for riadok in tactic:
            if 1 not in riadok: break
        try:
            if interesting_evaluation[(tactic[1], tactic[0], tactic[3], tactic[2])]: pass
        except KeyError:
            try:
                if interesting_evaluation[(tactic[3], tactic[2], tactic[1], tactic[0])]: pass
            except KeyError:
                interesting_evaluation[tactic] = True

    print(len(interesting_evaluation.keys()))
    return list(interesting_evaluation.keys())


import CHSHv00deterministic


def play_deterministic(tactic):
    """ Learns to play the best classic strategy according to tactic """
    env = CHSHv00deterministic.Environment(tactic)
    best = env.play_all_strategies()
    return best


import CHSHv02qDiscreteStatesActions


def play_quantum(evaluation_tactic):
    """ Learns to play the best quantum strategy according to tactic """
    ACTIONS2 = ['r' + axis + str(180 / 16 * i) for i in range(1, 3) for axis in 'xyz']
    ACTIONS = ['r' + axis + str(- 180 / 16 * i) for i in range(1, 3) for axis in 'xyz']
    ACTIONS2.extend(ACTIONS)
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2]
    ALL_POSSIBLE_ACTIONS.append("xxr0")
    ALL_POSSIBLE_ACTIONS.append("smallerAngle")
    ALL_POSSIBLE_ACTIONS.append("biggerAngle")

    N = 3000
    n_questions = 4
    max_gates = 9
    round_to = 2

    learning_rates = [0.1, 1, 0.01]
    gammas = [1, 0.9, 0.1]
    states = [np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.float64),
              np.array([1, 0, 0, 0], dtype=np.float64),
              np.array([0, 1 / sqrt(2), 1 / sqrt(2), 0], dtype=np.float64),
              np.array([0, 0, 1, 0], dtype=np.float64)]

    best = 0

    for state in states:
        env = CHSHv02qDiscreteStatesActions.Environment(n_questions, evaluation_tactic, max_gates,
                                                        initial_state=state)
        for alpha in learning_rates:
            for gamma in gammas:
                env.reset()
                # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
                agent = BasicAgent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=gamma, eps=1,
                                   eps_min=0.01,
                                   eps_decay=0.9995, alpha=alpha, momentum=0.9, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS,
                                   model_type=LinearModel)

                # scaler = get_scaler(env, N, ALL_POSSIBLE_ACTIONS, round_to=round_to)
                batch_size = 128

                # store the final value of the portfolio (end of episode)
                game = Game(round_to=round_to)
                portfolio_val = game.evaluate_train(N, agent, env)

                # save portfolio value for each episode
                np.save(f'.training/train.npy', portfolio_val)
                # portfolio_val = game.evaluate_test(agent, env)
                # return portfolio_val[0][0]  # acc

                load_acc = np.load(f'.training/train.npy')[0].max()
                if load_acc > best:
                    best = load_acc
    return best


def categorize(cutTactics):
    categories = dict()
    for tactic in cutTactics:
        classical_max = play_deterministic(tactic)
        if classical_max not in (0, 1):  # these are not interesting
            try:
                categories[classical_max].append(tactic)
            except KeyError:
                categories[classical_max] = [tactic]
    return categories


def max_entangled_difference(n):
    """ Prints evaluation tactics that had the biggest difference between classical and quantum strategy """
    categories = categorize(generate_only_interesting_tactics(n))

    differences = []
    for category, eval in categories.items():
        for _ in range(3):  # choose 10 tactics from each category randomly
            evaluation_tactic = random.choice(eval)
            classical_max = play_deterministic(evaluation_tactic)
            quantum_max = play_quantum(evaluation_tactic)
            difference_win_rate = 0 if classical_max > quantum_max else quantum_max - classical_max
            differences.append((category, evaluation_tactic, difference_win_rate))

    # differences.sort(key=lambda x: x[1])  # sorts according to difference in winning rate
    for category, evaluation_tactic, difference_win_rate in differences:
        print("category: ", category)
        print("evaluation_tactic = ")
        for i in evaluation_tactic: print(i)
        print("difference = ", difference_win_rate)


if __name__ == '__main__':
    # max_entangled_difference(size=4)
    # evaluation_tactic = [[1, 0, 0, 1],
    #                      [1, 0, 0, 1],
    #                      [1, 0, 0, 1],
    #                      [0, 1, 1, 0]]
    # print(play_deterministic(evaluation_tactic))

    # print(len(generate_only_interesting_tactics(4)))

    max_entangled_difference(4)
