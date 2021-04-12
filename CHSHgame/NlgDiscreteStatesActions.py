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
        self.n_games = n_games

        self.n_questions = n_questions
        self.n_players = n_players
        self.counter = 1
        self.history_actions = []
        self.history_actions_anneal = []

        self.max_gates = max_gates
        self.min_gates = 0
        self.game_type = game_type
        self.initial_state = initial_state
        self.state = self.initial_state.copy()

        self.n_qubits = self.n_qubits_from_state()
        self.reduce_by = 2 ** (self.n_qubits - 2)

        self.possible_states = list(
            itertools.product(list(range(self.n_questions)),
                              repeat=self.n_qubits))

        self.one_game_answers = list(
            itertools.product(list(range(self.n_questions)),
                              repeat=self.n_players))


        self.repr_state = np.array([x for _ in range(self.n_players ** 2) for x in self.state], dtype=np.complex64)

        self.state_size = len(self.repr_state) * 2  # times 2 because of complex array to array of real numbers

        self.accuracy = self.calc_accuracy([self.measure_analytic() for _ in range(self.n_questions * self.n_players)])
        self.max_acc = self.accuracy
        self.min_acc = self.accuracy

        self.max_found_state = self.repr_state.copy()
        self.max_found_strategy = []
        self.min_found_state = self.repr_state.copy()
        self.min_found_strategy = []
        self.best_or_worst = best_or_worst

        self.questions = list(itertools.product(list(range(self.n_questions)), repeat=self.n_players))
        print(self.questions)
        self.memory_state = dict()
        self.velocity = 1
        self.reward_funcion = reward_function
        if self.reward_funcion == None: self.reward_funcion = self.reward_only_difference

        self.use_annealing = anneal

    @NonLocalGame.override
    def reset(self):
        self.velocity = 1
        self.history_actions_anneal = []
        return self.complex_array_to_real(super().reset())  # + np.array([len(self.history_actions)], dtype=np.float64)

    def calculate_state(self, history_actions, anneal=False):
        """ Calculates the state according to previous actions"""
        result = []

        for g, q in enumerate(self.questions):
            # Alice - a and Bob - b share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input, Bob too - eg. a0 if alice gets 0 as input

            self.state = self.initial_state.copy()
            self.velocity = 1

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
                try: gate_angle = np.array([action[4:]], dtype=np.float32) * self.velocity
                except ValueError: gate_angle = 0

                I_length = int(len(self.initial_state) ** (1 / self.n_players))

                # apply action to state
                operation = []

                if gate == CXGate:
                    if to_whom in 'a0a1':
                        ctrl = int(action[-1] != "r")
                        operation = np.kron(CXGate(ctrl_state=ctrl).to_matrix(), np.identity(I_length))
                    elif to_whom in 'b0b1':
                        ctrl = int(action[-1] == "r")
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
            self.max_found_state = self.repr_state.copy()
            self.max_found_strategy = self.history_actions_anneal.copy()

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

        try:
            result, self.repr_state = self.memory_state[tuple(self.history_actions)][:2]
        except KeyError:
            try: result, self.repr_state, self.history_actions_anneal[:-1] = self.memory_state[tuple(self.history_actions[:-1])]
            except: pass
            if action not in {"xxr0", "smallerAngle", "biggerAngle", "a0cxnot", "b0cxnot", "a0cxnotr", "b0cxnotr"} and self.use_annealing:
                self.history_actions_anneal[-1] = self.history_actions_anneal[-1][:4] + str(
                    self.anneal())  # simulated annealing on the last chosen action

            if self.use_annealing: result = self.calculate_state(self.history_actions_anneal)
            else: result = self.calculate_state(self.history_actions)
            self.memory_state[tuple(self.history_actions)] = (result, self.repr_state.copy(), self.history_actions_anneal.copy())

        # accuracy of winning CHSH game
        before = self.accuracy
        self.accuracy = self.calc_accuracy(result)

        difference_in_accuracy = self.accuracy - before

        if self.best_or_worst == "worst": difference_in_accuracy *= (-1)

        try:
            reward = self.reward_funcion(self, difference_in_accuracy)
        except:
            reward = self.reward_funcion(difference_in_accuracy)

        self.save_interesting_strategies()

        if self.counter == self.max_gates or action == 'xxr0': done = True
        if not done: self.counter += 1
        return self.complex_array_to_real(self.repr_state), reward, done

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

if __name__ == '__main__':
    # Hyperparameters setting
    # ACTIONS2 = ['r' + axis + str(180 / 32 * i) for i in range(1, 16) for axis in 'y']
    # ACTIONS = ['r' + axis + str(-180 / 32 * i) for i in range(1, 16) for axis in 'y']
    ACTIONS2 = [q + axis + "0" for axis in 'xyz' for q in 'ra']
    # ACTIONS2 = [q + axis + "0" for axis in 'y' for q in 'r']
    # ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [[p + q + a] for p in PERSON for q in QUESTION for a in ACTIONS2]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append(["xxr0"])
    # ALL_POSSIBLE_ACTIONS.append("smallerAngle")
    # ALL_POSSIBLE_ACTIONS.append("biggerAngle")

    # for 1 game with 2 EPR
    ALL_POSSIBLE_ACTIONS.append(["a0cxnot"])
    ALL_POSSIBLE_ACTIONS.append(["b0cxnot"])

    # for xor paralel with 2EPR
    ALL_POSSIBLE_ACTIONS.append(["a0cxnotr"])
    ALL_POSSIBLE_ACTIONS.append(["b0cxnotr"])

    N = 4000
    n_questions = 2
    game_type = [[1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [0, 1, 1, 0]]
    max_gates = 10
    round_to = 6
    state = np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex64)
    state = np.array(
        [0 + 0j, 0 + 0j, 0 + 0j, 0.5 + 0j, 0 + 0j, -0.5 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, -0.5 + 0j, 0 + 0j, 0.5 + 0j, 0 + 0j, 0 + 0j, 0 + 0j], dtype=np.complex64)
    env = Environment(n_questions, game_type, max_gates, initial_state=state, reward_function=Environment.reward_only_difference, anneal=True,
                      n_games=2)

    hidden_dim = [len(env.repr_state) * 2, len(env.repr_state) * 2, len(env.repr_state) // 2, len(env.repr_state)]

    # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
    # agent = BasicAgent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=1, eps=1, eps_min=0.01,
    #                    eps_decay=0.9995, alpha=0.001, momentum=0.9, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS,
    #                    model_type=LinearModel)

    # define one hot encoding
    encoder = OneHotEncoder(drop='first', sparse=False)
    # transform data
    onehot = encoder.fit_transform(ALL_POSSIBLE_ACTIONS)

    onehot_to_action = dict()
    action_to_onehot = dict()
    for x, a_encoded in enumerate(onehot):
        onehot_to_action[str(a_encoded)] = x
        action_to_onehot[x] = str(a_encoded)

    agent = DQNAgent(state_size=env.state_size, action_size=len(ALL_POSSIBLE_ACTIONS), gamma=0.1, eps=1, eps_min=0.01,
                     eps_decay=0.9998, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, learning_rate=0.001, hidden_layers=len(hidden_dim),
                     hidden_dim=hidden_dim, onehot_to_action=onehot_to_action, action_to_onehot=action_to_onehot)

    # scaler = get_scaler(env, N**2, ALL_POSSIBLE_ACTIONS, round_to=round_to)
    # The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset.
    # The number of epochs can be set to an integer value between one and infinity.
    batch_size = 128

    # store the final value of the portfolio (end of episode)
    game = Game(round_to=round_to, batch_size=batch_size)
    portfolio_value, rewards = game.evaluate_train(N, agent, env)

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
