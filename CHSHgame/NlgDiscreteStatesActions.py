import itertools
import random
from math import sqrt, pi

import numpy as np
from qiskit.circuit.library import IGate, CXGate
from sklearn.preprocessing import OneHotEncoder

import NonLocalGame
from NonLocalGame import Game
from agents.DQNAgent import DQNAgent


class Environment(NonLocalGame.abstractEnvironment):
    """ Creates CHSH environments for quantum strategies, discretizes and states and uses discrete actions """

    def __init__(self, n_questions, game_type, max_gates, n_players=2,
                 initial_state=np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex64), best_or_worst="best", reward_function=None,
                 anneal=False, n_games=1):
        self.n_games = n_games  # how many games are to be played (paralel)
        self.n_questions = n_questions  # how many atomic questions (to one player)
        self.n_players = n_players # players / verifiers
        self.counter = 1   # number of steps
        self.history_actions = []  # history of actions taken
        self.history_actions_anneal = [] # history of action annealed taken

        self.max_gates = max_gates # limit of gates that can be taken
        self.min_gates = 0
        self.game_type = game_type  # game matrix (rules - when do they win)
        self.initial_state = initial_state  # initial state
        self.state = self.initial_state.copy()

        self.n_qubits = self.n_qubits_from_state()
        self.reduce_by = 2 ** (self.n_qubits - 2)  # reducing for double games

        self.possible_states = list(   # possible states
            itertools.product(list(range(self.n_questions)),
                              repeat=self.n_qubits))

        self.one_game_answers = list(  # possible answers
            itertools.product(list(range(self.n_questions)),
                              repeat=self.n_players))

        self.repr_state = np.array([x for _ in range(len(self.game_type)) for x in self.state], dtype=np.complex64) # state representation for all comb. of questions

        self.state_size = len(self.repr_state) * 2  # times 2 because of complex array to array of real numbers

        self.accuracy = self.calc_accuracy([self.measure_analytic() for _ in range(len(self.game_type))]) # winning probability
        self.max_acc = self.accuracy
        self.min_acc = self.accuracy

        self.max_found_state = self.repr_state.copy()   # best / worst found configurations
        self.max_found_strategy = []
        self.min_found_state = self.repr_state.copy()
        self.min_found_strategy = []
        self.best_or_worst = best_or_worst

        self.questions = list(itertools.product(list(range(self.n_questions)), repeat=self.n_players*self.n_games)) # combinations of questions
        print(self.questions)
        self.memory_state = dict() # memoization of calculation, repr_state, accuracies
        self.reward_funcion = reward_function
        if self.reward_funcion == None: self.reward_funcion = self.reward_only_difference

        self.immutable = {"xxr0", "smallerAngle", "biggerAngle", "a0cxnot", "b0cxnot", "a0cxnotr", "b0cxnotr"}

        self.use_annealing = anneal # do you want to use annealing?

    @NonLocalGame.override
    def reset(self):
        self.history_actions_anneal = []
        return self.complex_array_to_real(super().reset())  # + np.array([len(self.history_actions)], dtype=np.float64)

    def calculate_state(self, history_actions, anneal=False):
        """ Calculates the state according to previous actions in parameter history_actions """
        result = []

        for g, q in enumerate(self.questions):
            # Alice - a and Bob - b share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input, Bob too - eg. a0 if alice gets 0 as input

            self.state = self.initial_state.copy()

            for action in history_actions:
                # get info from action
                # if action == "biggerAngle":
                #     self.velocity *= 2
                #     continue
                # elif action == "smallerAngle":
                #     self.velocity /= 2
                #     continue

                # decode action
                gate = self.get_gate(action)
                if gate == IGate: continue
                to_whom = action[0:2]
                rotate_ancilla = action[2] == 'a'
                try: gate_angle = np.array([action[4:]], dtype=np.float32)
                except ValueError: gate_angle = 0

                I_length = int(len(self.initial_state) ** (1 / self.n_players))

                # apply action to state
                operation = []

                if gate == CXGate:
                    ctrl = int(action[-1] != "r")
                    if (q[0] == 0 and to_whom == 'a0') or (q[0] == 1 and to_whom == 'a1'):
                        operation = np.kron(CXGate(ctrl_state=ctrl).to_matrix(), np.identity(I_length))
                    if (q[1] == 0 and to_whom == 'b0') or (q[1] == 1 and to_whom == 'b1'):
                        operation = np.kron(np.identity(I_length), CXGate(ctrl_state=ctrl).to_matrix())
                else:
                    if (q[0] == 0 and to_whom == 'a0') or (q[0] == 1 and to_whom == 'a1'):
                        if rotate_ancilla:  calc_operation = np.kron(gate((gate_angle * pi / 180).item()).to_matrix(), np.identity(2))
                        else: calc_operation = np.kron(np.identity(2), gate((gate_angle * pi / 180).item()).to_matrix())
                        if len(self.state) != 4: operation = np.kron(calc_operation, np.identity(I_length))
                        else: operation = calc_operation
                    if (q[1] == 0 and to_whom == 'b0') or (q[1] == 1 and to_whom == 'b1'):
                        if rotate_ancilla:  calc_operation = np.kron(np.identity(2), gate((gate_angle * pi / 180).item()).to_matrix())
                        else: calc_operation = np.kron(gate((gate_angle * pi / 180).item()).to_matrix(), np.identity(2))
                        if len(self.state) != 4: operation = np.kron(np.identity(I_length), calc_operation)
                        else: operation = calc_operation

                if len(operation) != 0:
                    self.state = np.matmul(operation, self.state)

            # modify repr_state according to state
            self.repr_state[g * len(self.state):(g + 1) * len(self.state)] = self.state.copy()

            result.append(self.measure_analytic())

        return result

    def save_interesting_strategies(self):
        if self.accuracy > self.max_acc:
            self.max_acc = self.accuracy
            self.max_found_state = self.repr_state.copy()
            self.max_found_strategy = self.history_actions_anneal.copy()

        elif self.accuracy == self.max_acc:
            if len(self.history_actions) < len(self.max_found_strategy):
                self.max_found_state = self.repr_state.copy()
                self.max_found_strategy = self.history_actions_anneal.copy()

        if self.accuracy < self.min_acc:
            self.min_acc = self.accuracy
            self.min_found_state = self.repr_state.copy()
            self.min_found_strategy = self.history_actions_anneal.copy()

        elif self.accuracy == self.min_acc:
            if len(self.history_actions) < len(self.min_found_strategy):
                self.min_found_state = self.repr_state.copy()
                self.min_found_strategy = self.history_actions_anneal.copy()

        if self.min_found_strategy == []: self.min_found_strategy.append('xxr0')
        if self.max_found_strategy == []: self.max_found_strategy.append('xxr0')

    @NonLocalGame.override
    def step(self, action):
        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        if type(action) == list: action = action[0]
        # play game
        self.history_actions.append(action)
        self.history_actions_anneal.append(action)

        # accuracy of winning CHSH game
        before = self.accuracy

        try:
            result, self.repr_state, _, self.accuracy, to_complex = self.memory_state[tuple(self.history_actions)]
        except KeyError:
            try: result, self.repr_state, self.history_actions_anneal[:-1] = self.memory_state[tuple(self.history_actions[:-1])][:3]
            except KeyError: pass
            if action not in self.immutable and self.use_annealing:
                self.history_actions_anneal[-1] = self.history_actions_anneal[-1][:4] + str(
                    self.anneal())  # simulated annealing on the last chosen action

            if self.use_annealing: result = self.calculate_state(self.history_actions_anneal)
            else: result = self.calculate_state(self.history_actions)

            self.accuracy = self.calc_accuracy(result)
            to_complex = self.complex_array_to_real(self.repr_state)
            self.memory_state[tuple(self.history_actions)] = (
                result, self.repr_state.copy(), self.history_actions_anneal.copy(), self.accuracy, to_complex)

        difference_in_accuracy = self.accuracy - before

        if self.best_or_worst == "worst": difference_in_accuracy *= (-1)

        try: reward = self.reward_funcion(self, difference_in_accuracy)  # because I needed to call like this when using Optimalizing hyperparam.
        except: reward = self.reward_funcion(difference_in_accuracy)

        self.save_interesting_strategies()

        if self.counter == self.max_gates or action == 'xxr0': done = True
        if not done: self.counter += 1
        return to_complex, reward, done

    def anneal(self, steps=100, t_start=2, t_end=0.001):
        """ Finds the maximal value of the fitness function by
        executing the simulated annealing algorithm.
        Returns a state (e.g. x) for which fitness(x) is maximal. """
        x = self.random_state()
        t = t_start
        for i in range(steps):
            neighbor = np.random.choice(self.neighbors(x))
            ΔE = self.fitness(neighbor) - self.fitness(x)
            if ΔE > 0:  # //neighbor is better then x
                x = neighbor
            elif np.random.random() < np.math.e ** (ΔE / t):  # //neighbor is worse then x
                x = neighbor
            t = t_start * (t_end / t_start) ** (i / steps)
        return x

    def fitness(self, x):
        """ Calculates fitness of the state given by calculation of accuracy over history of actions."""
        last = [self.history_actions_anneal[-1][:4] + str(x)]
        return self.calc_accuracy(self.calculate_state(self.history_actions_anneal[:-1] + last, anneal=True))

    def neighbors(self, x, span=30, delta=0.1):
        """ Creates neighboring gate angle to angle x"""
        res = []
        if x > -span + 3 * delta: res += [x - i * delta for i in range(1, 4)]
        if x < span - 3 * delta: res += [x + i * delta for i in range(1, 4)]
        return res

    def random_state(self):
        return random.uniform(-180, 180)


import warnings

warnings.filterwarnings('ignore')
from NonLocalGame import show_plot_of
from NlgDeterministic import create

if __name__ == '__main__':
    # Hyperparameters setting
    # ACTIONS = [q + axis + "0" for axis in 'xyz' for q in 'ra']
    ACTIONS = [q + axis + "0" for axis in 'y' for q in 'r']
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [[p + q + a] for p in PERSON for q in QUESTION for a in ACTIONS]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append(["xxr0"])

    # # for 1 game with 2 EPR
    # ALL_POSSIBLE_ACTIONS.append(["a0cxnot"])
    # ALL_POSSIBLE_ACTIONS.append(["b0cxnot"])
    #
    # # for xor paralel with 2EPR
    # ALL_POSSIBLE_ACTIONS.append(["a0cxnotr"])
    # ALL_POSSIBLE_ACTIONS.append(["b0cxnotr"])

    N = 4000
    n_questions = 2
    game_type = [[1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [0, 1, 1, 0]]

    max_gates = 10
    round_to = 6
    state = np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex64)
    env = Environment(n_questions, game_type, max_gates, initial_state=state,
                      reward_function=Environment.reward_only_difference,
                      anneal=True, n_games=1)



    # transform actions to noncorellated encoding
    encoder = OneHotEncoder(drop='first', sparse=False)
    # transform data
    onehot = encoder.fit_transform(ALL_POSSIBLE_ACTIONS)
    onehot_to_action = dict()
    action_to_onehot = dict()
    for x, a_encoded in enumerate(onehot):
        onehot_to_action[str(a_encoded)] = x
        action_to_onehot[x] = str(a_encoded)

    hidden_dim = [len(env.repr_state) * 2, len(env.repr_state) * 2, len(env.repr_state) // 2, len(env.repr_state)]
    agent = DQNAgent(state_size=env.state_size, action_size=len(ALL_POSSIBLE_ACTIONS), gamma=1, eps=1, eps_min=0.01,
                     eps_decay=0.9998, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, learning_rate=0.001, hidden_layers=len(hidden_dim),
                     hidden_dim=hidden_dim, onehot_to_action=onehot_to_action, action_to_onehot=action_to_onehot)
    # divide data by
    batch_size = 128

    game = Game(round_to=round_to, batch_size=batch_size)
    portfolio_value, rewards = game.evaluate_train(N, agent, env)

    # agent = DQNAgent(state_size=env.state_size, action_size=len(ALL_POSSIBLE_ACTIONS), gamma=1, eps=1, eps_min=0.01,
    #                  eps_decay=0.9998, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, learning_rate=0.001, hidden_layers=len(hidden_dim),
    #                  hidden_dim=hidden_dim, onehot_to_action=onehot_to_action, action_to_onehot=action_to_onehot)

    # The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset.


    # plot relevant information
    show_plot_of(rewards, "reward")

    if agent.model.losses is not None:
        show_plot_of(agent.model.losses, "loss")

    show_plot_of(portfolio_value, "accuracy", [0.85, 0.75])

    # save portfolio value for each episode
    np.save(f'.training/train.npy', portfolio_value)

    portfolio_value = game.evaluate_test(agent, env)
    print(portfolio_value)
    a = np.load(f'.training/train.npy')
    print(f"average accuracy: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")
