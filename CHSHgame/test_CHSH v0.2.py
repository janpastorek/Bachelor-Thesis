import unittest
from CHSHv02 import Environment
import numpy as np
from qiskit.extensions import RYGate
from math import pi

class TestCHSH(unittest.TestCase):

    def testOne(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [*env.state,*env.state,*env.state,*env.state]

        win_rate = 0
        for mat in result[:-1]:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        win_rate += 1 / 4 * (result[-1][1] + result[-1][2])


        assert (False)

    def testMaxCHSH(self):
        # toto sa naucil
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        save_state = env.initial_state.copy()
        nauceneVyhodil = ['b0r-78.75', 'b0r-78.75', 'a0r90.0', 'b0r-78.75', 'b1r56.25', 'b1r-22.5', 'b0r11.25', 'b1r0.0', 'b1r0.0', 'b1r0.0']
        dokopy=['a0r90','b0r236.25','b1r33.75']
        for a in dokopy:
            env.step(a)
        print(env.repr_state)
        print(np.array([*np.matmul(np.kron(RYGate((90 * pi / 180)).to_matrix(), np.identity(2)), save_state),
                                  *save_state,
                                  *np.matmul(np.kron(RYGate((236.25 * pi / 180)).to_matrix(), np.identity(2)), save_state),
                                  *np.matmul(np.kron(RYGate((33.75 * pi / 180)).to_matrix(), np.identity(2)), save_state)]))

        assert(env.repr_state.all() == np.array([np.matmul(np.kron(RYGate((90 * pi / 180)).to_matrix(), np.identity(2)), save_state),
                                  save_state,
                                  np.matmul(np.kron(RYGate((236.25 * pi / 180)).to_matrix(), np.identity(2)), save_state),
                                  np.matmul(np.kron(RYGate((33.75 * pi / 180)).to_matrix(), np.identity(2)), save_state)]).all())




if __name__=="__main__":
    unittest.main()