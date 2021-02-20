import unittest
from CHSHv02quantumDiscreteStatesActions import Environment
from CHSHv05quantumGeneticOptimalization import CHSHgeneticOptimizer
import numpy as np
from qiskit.extensions import RYGate
from math import pi


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

    def testGeneticAlg(self):
        history_actions = ['a0r0', 'b0r0', 'a1r0', 'b1r0']
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        ga = CHSHgeneticOptimizer(population_size=15, n_crossover=3, mutation_prob=0.05, history_actions=history_actions,
                                  evaluation_tactic=tactic)
        best = ga.solve(50)  # you can also play with max. generations
        assert best[1] >= 0.83

    def testTensorflow(self):
        pass
        # import numpy as np
        #
        # from tensorflow.keras import layers, models
        #
        # IMAGE_WIDTH = 128
        # IMAGE_HEIGHT = 128
        #
        # model = models.Sequential()
        # # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
        # # model.add(layers.MaxPooling2D((2, 2)))
        # # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # # model.add(layers.MaxPooling2D((2, 2)))
        # # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # # model.add(layers.Flatten())
        # model.add(layers.Dense(4, activation='relu'))
        # model.add(layers.Dense(32, activation='softmax'))
        #
        # model.compile(optimizer='adam',
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        #
        # BATCH_SIZE = 128
        #
        # images = np.zeros((BATCH_SIZE, IMAGE_WIDTH))
        # labels = np.zeros((BATCH_SIZE, 32))
        #
        # history = model.fit(images, labels, epochs=1)

    def testTensorflow1(self):
        import tensorflow as tf
        hello = tf.constant("hello TensorFlow!")

    def testCHSHdeterministicStrategies(self):
        import CHSH
        evaluation_tactic = [[1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 1, 0]]
        assert CHSH.play_deterministic(evaluation_tactic) == 0.75

    def testCHSHacc(self):
        import CHSHv02quantumDiscreteStatesActions
        naucil_sa = ['b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'biggerAngle', 'a0ry22.5', 'b1ry-22.5']
        dokopy = ['bory-135', 'a0ry45', 'b1ry-45']

        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        env = CHSHv02quantumDiscreteStatesActions.Environment(n_questions=4,evaluation_tactic=tactic,max_gates=10)
        for a in dokopy:
            env.step(a)

        print(env.accuracy)
        assert env.accuracy < 0.86




if __name__ == "__main__":
    unittest.main()
