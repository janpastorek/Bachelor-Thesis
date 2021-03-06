import itertools
from math import sqrt, pi

import numpy as np
from qiskit.circuit.library import IGate, CXGate

import CHSH
from CHSH import Game
from agents.DQNAgent import DQNAgent


class Environment(CHSH.abstractEnvironment):
    """ Creates CHSH environments for quantum strategies, discretizes and states and uses discrete actions """

    def __init__(self, n_questions, game_type, max_gates, num_players=2,
                 initial_state=np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.float64), best_or_worst="best"):
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = []
        self.max_gates = max_gates
        self.game_type = game_type
        self.initial_state = initial_state
        self.state = self.initial_state.copy()
        self.num_players = num_players
        self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state], dtype=np.float64)
        self.accuracy = self.calc_accuracy([self.measure_analytic() for _ in range(self.n_questions)])
        self.max_acc = self.accuracy
        # input, generate "questions" in equal number
        # self.a = []
        # self.b = []

        self.best_or_worst = best_or_worst

        self.questions = list(itertools.product(list(range(self.n_questions // 2)), repeat=self.num_players))
        print(self.questions)
        # for x in range(2):
        #     for y in range(2):
        #         self.a.append(x)
        #         self.b.append(y)

        self.memory_state = dict()

        self.velocity = 1

    @CHSH.override
    def reset(self):
        self.velocity = 1
        return super().reset()  # + np.array([len(self.history_actions)], dtype=np.float64)

    def calculate_state(self, history_actions):
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
                rotate_ancilla = action[2] == 'a'
                try:
                    gate_angle = np.array([action[4:]], dtype=np.float64) * self.velocity
                except ValueError:
                    gate_angle = 0

                I_length = int(len(self.initial_state) ** (1 / self.num_players))

                # apply action to state
                operation = []
                if gate == CXGate:
                    if to_whom in 'a0a1':
                        operation = np.kron(CXGate(ctrl_state=1).to_matrix(), np.identity(I_length))
                    elif to_whom in 'b0b1':
                        operation = np.kron(np.identity(I_length), CXGate(ctrl_state=0).to_matrix())
                else:
                    if (q[0] == 0 and to_whom == 'a0') or (q[0] == 1 and to_whom == 'a1'):
                        if rotate_ancilla: calc_operation = np.kron(gate((gate_angle * pi / 180).item()).to_matrix(), np.identity(2))
                        else: calc_operation = np.kron(np.identity(2), gate((gate_angle * pi / 180).item()).to_matrix())
                        if len(self.state) != 4: operation = np.kron(calc_operation, np.identity(I_length))
                        else: operation = calc_operation
                    if (q[1] == 0 and to_whom == 'b0') or (q[1] == 1 and to_whom == 'b1'):
                        if rotate_ancilla: calc_operation = np.kron(np.identity(2), gate((gate_angle * pi / 180).item()).to_matrix())
                        else: calc_operation = np.kron(gate((gate_angle * pi / 180).item()).to_matrix(), np.identity(2))
                        if len(self.state) != 4: operation = np.kron(np.identity(I_length), calc_operation)
                        else: operation = calc_operation

                if operation != []:
                    self.state = np.matmul(operation, self.state)
                    operation = []

            # modify repr_state according to state
            self.repr_state[g * len(self.state):(g + 1) * len(self.state)] = self.state.copy()  # TODO: co tam teraz ulozit?

            result.append(self.measure_analytic())  # TODO: ako sa bude teraz meriat win_rate? treba zistit, ci a na ktorom subarray treba zmerat

        # self.repr_state[-1] = len(self.history_actions)
        self.memory_state[tuple(history_actions)] = (result, self.repr_state)
        return result

    @CHSH.override
    def step(self, action):
        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        # play game
        self.history_actions.append(action)
        try:
            result, self.repr_state = self.memory_state[tuple(self.history_actions)]
        except KeyError:
            result = self.calculate_state(self.history_actions)

        # accuracy of winning CHSH game
        before = self.accuracy
        self.accuracy = self.calc_accuracy(result)

        # reward is the increase in accuracy
        rozdiel_acc = self.accuracy - before
        reward = rozdiel_acc * 100

        # ends when it has applied max number of gates / xxr0
        if np.round(self.accuracy, 2) >= np.round(self.max_acc, 2):
            done = True
            self.max_acc = self.accuracy
            reward += 50 * (1 / (self.count_gates() + 1))

        if self.counter == self.max_gates or self.history_actions[-1] == 'xxr0':
            done = True
            if np.round(self.max_acc, 2) == np.round(self.accuracy, 2):
                reward += 200 * (1 / (self.count_gates() + 1))

        if self.best_or_worst == "worst": reward *= (-1)
        if done: reward -= self.count_gates() / self.accuracy

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
from CHSH import show_plot_of

if __name__ == '__main__':
    # Hyperparameters setting
    ACTIONS2 = ['r' + axis + str(360 / 4 * i) for i in range(1, 2) for axis in 'xyz']
    ACTIONS = ['r' + axis + str(-360 / 4 * i) for i in range(1, 2) for axis in 'xyz']
    ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append("xxr0")
    ALL_POSSIBLE_ACTIONS.append("smallerAngle")
    ALL_POSSIBLE_ACTIONS.append("biggerAngle")
    ALL_POSSIBLE_ACTIONS.append("cnot")  # can be used only when state is bigger than 4

    N = 2000
    n_questions = 4
    evaluation_tactic = [[1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0]]
    max_gates = 10
    round_to = 3
    env = Environment(n_questions, evaluation_tactic, max_gates, initial_state=np.array(
        [0 + 0j, 0 + 0j, 0.707 + 0j, 0 + 0j, -0.707 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j]))
    hidden_dim = [len(env.repr_state)]

    # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
    # agent = BasicAgent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=1, eps=1, eps_min=0.01,
    #                    eps_decay=0.9995, alpha=0.001, momentum=0.9, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS,
    #                    model_type=LinearModel)

    agent = DQNAgent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=0.9, eps=1, eps_min=0.01,
                     eps_decay=0.9995, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, learning_rate=0.001, hidden_layers=len(hidden_dim),
                     hidden_dim=hidden_dim)

    # scaler = get_scaler(env, N**2, ALL_POSSIBLE_ACTIONS, round_to=round_to)
    batch_size = 128

    # store the final value of the portfolio (end of episode)
    game = Game(round_to=round_to)
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
    print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")
