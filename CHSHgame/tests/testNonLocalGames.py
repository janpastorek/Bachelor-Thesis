import unittest
from NlgDiscreteStatesActions import Environment
from NlgGeneticOptimalization import CHSHgeneticOptimizer
import numpy as np
from qiskit.extensions import RYGate
from math import pi, sqrt


class TestCHSH(unittest.TestCase):

    def testRYGate(self):
        assert (np.around(RYGate((0 * pi / 180)).to_matrix(), 5).all() == np.eye(2).all())

    def testIfCorrectStrategyAndAccuracy(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        save_state = env.initial_state.copy()
        nauceneVyhodil = ['b0r-78.75', 'b0r-78.75', 'a0r90.0', 'b0r-78.75', 'b1r56.25', 'b1r-22.5', 'b0r11.25',
                          'b1r0.0', 'b1r0.0', 'b1r0.0']  # toto sa naucil
        dokopy = ['a0ry90', 'b0ry-225', 'b1ry33.75']
        for a in dokopy:
            env.step(a)

        A_0 = np.kron(RYGate((90 * pi / 180)).to_matrix(), np.identity(2))
        A_1 = np.kron(np.identity(2), np.identity(2))
        B_0 = np.kron(np.identity(2), RYGate((-225 * pi / 180)).to_matrix())
        B_1 = np.kron(np.identity(2), RYGate((33.75 * pi / 180)).to_matrix())

        ax = np.array([
            *[x for x in np.matmul(B_0, np.matmul(A_0, save_state))],
            *[x for x in np.matmul(B_1, np.matmul(A_0, save_state))],
            *[x for x in np.matmul(B_0, np.matmul(A_1, save_state))],
            *[x for x in np.matmul(B_1, np.matmul(A_1, save_state))]
        ])
        print(ax)
        print(env.accuracy)
        # assert (env.accuracy > 0.85) //TODO: este raz prekontrolovat ci je to spravne
        for poc, state in enumerate(env.repr_state):
            if poc % 4 == 0:
                assert (np.round(
                    env.repr_state[poc] ** 2 + env.repr_state[poc + 1] ** 2 + env.repr_state[poc + 2] ** 2 +
                    env.repr_state[poc + 3] ** 2, 2) == 1)

        for poc, state in enumerate(ax):
            if poc % 4 == 0:
                assert (np.round(ax[poc] ** 2 + ax[poc + 1] ** 2 + ax[poc + 2] ** 2 + ax[poc + 3] ** 2, 2) == 1)

        assert (env.repr_state.all() == ax.all())

    def testInitialAccuracy(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        assert (np.round(env.accuracy,2) == 0.25)

    # check if the other way of calculating accuracy is correct through comparing with already known good way, but inflexible
    def testCalcWinRate(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_analytic() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result[:-1]:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        win_rate += 1 / 4 * (result[-1][1] + result[-1][2])
        assert (win_rate == env.calc_accuracy(result))

    def testCalcWinRate1(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_analytic() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        assert (win_rate == env.calc_accuracy(result))

    def testCalcWinRate2(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 0, 1]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_analytic() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result[:-1]:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        win_rate += 1 / 4 * (result[-1][1] + result[-1][3])

        assert (win_rate == env.calc_accuracy(result))

    def testCalcWinRate3(self):
        n_questions = 4
        tactic = [[1,1,1,1] for i in range(n_questions)]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_analytic() for i in range(4)]
        assert (round(env.calc_accuracy(result)) == 1)

    def testCalcWinRate4(self):
        n_questions = 4
        tactic = [[0,0,0,0] for i in range(n_questions)]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_analytic() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result[:-1]:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        win_rate += 1 / 4 * (result[-1][1] + result[-1][3])

        assert round(win_rate - env.calc_accuracy(result) - 0) == 0

    def testGeneticAlg(self):
        # Solve to find optimal individual
        ACTIONS2 = ['r' + axis + "0" for axis in 'xyz']
        # ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
        PERSON = ['a', 'b']
        QUESTION = ['0', '1']

        ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2]  # place one gate at some place
        game = [[1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0]]
        ga = CHSHgeneticOptimizer(population_size=30, n_crossover=len(ALL_POSSIBLE_ACTIONS) - 1, mutation_prob=0.1,
                                  history_actions=ALL_POSSIBLE_ACTIONS,
                                  game_type=game, best_or_worst="best", state=np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex128))

        best = ga.solve(22)  # you can also play with max. generations
        ga.show_individual(best[0])
        assert best[1] >= 0.83

    def testGeneticAlg2(self):
        # Solve to find optimal individual
        ACTIONS2 = ['r' + axis + "0" for axis in 'y']
        # ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
        PERSON = ['a', 'b']
        QUESTION = ['0', '1']

        ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2]  # place one gate at some place
        game = [[0, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 1, 0, 0]]
        ga = CHSHgeneticOptimizer(population_size=30, n_crossover=len(ALL_POSSIBLE_ACTIONS) - 1, mutation_prob=0.1,
                                  history_actions=ALL_POSSIBLE_ACTIONS,
                                  game_type=game, best_or_worst="best", state=np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex128))
        best = ga.solve(22)  # you can also play with max. generations
        ga.show_individual(best[0])
        assert np.round(best[1],2) == 0.5

    def testTensorflow1(self):
        import tensorflow as tf
        hello = tf.constant("hello TensorFlow!")

    def testCHSHdeterministicStrategies(self):
        import NonLocalGame
        evaluation_tactic = [[1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 1, 0]]
        assert NonLocalGame.play_deterministic(evaluation_tactic)[0] == 0.75
        assert NonLocalGame.play_deterministic(evaluation_tactic)[1] == 0.25

    def testCHSHacc(self):
        import NlgDiscreteStatesActions
        naucil_sa = ['b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'biggerAngle', 'a0ry22.5', 'b1ry-22.5']
        dokopy = ['b0ry-135', 'a0ry45', 'b1ry-45']

        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        env = NlgDiscreteStatesActions.Environment(n_questions=4, game_type=tactic, max_gates=10)
        for a in dokopy:
            env.step(a)

        print(env.accuracy)
        assert env.accuracy < 0.86

    def testCHSH2epr(self):
        import NlgDiscreteStatesActions

        max_gates = 10
        n_questions = 2
        game_type = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        state = np.array(
            [0 + 0j, 0 + 0j, 0 + 0j, 0.5 + 0j, 0 + 0j, -0.5 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, -0.5 + 0j, 0 + 0j, 0.5 + 0j, 0 + 0j, 0 + 0j,
             0 + 0j], dtype=np.complex64)
        env = Environment(n_questions, game_type, max_gates, initial_state=state, reward_function=Environment.reward_only_difference, anneal=True,
                          n_games=1)
        assert np.round(env.accuracy,2) == 0.5



if __name__ == "__main__":
    unittest.main()
