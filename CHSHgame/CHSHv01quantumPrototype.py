import numpy as np
import random

from qiskit.circuit.library import RYGate


def is_unitary(M):
    M_star = np.transpose(M).conjugate()
    identity = np.eye(len(M))
    return np.allclose(identity, np.matmul(M_star, M))


class QuantumState:
    def __init__(self, vector):
        length = np.linalg.norm(vector)
        if not abs(1 - length) < 0.00001:
            raise ValueError('Quantum states must be unit length.')
        self.vector = np.array(vector)

    def measure(self):
        choices = range(len(self.vector))
        weights = [abs(a) ** 2 for a in self.vector]
        print(weights)
        outcome = random.choices(choices, weights)[0]

        new_state = np.zeros(len(self.vector))
        new_state[outcome] = 1
        self.vector = new_state
        return outcome

    def measure_analytic(self):
        choices = range(len(self.vector))
        weights = [abs(a) ** 2 for a in self.vector]

        return weights

    def compose(self, state):
        new_vector = np.kron(self.vector, state.vector)
        return QuantumState(new_vector)

    def __repr__(self):
        return '<QuantumState: {}>'.format(', '.join(map(str, self.vector)))


class QuantumOperation:
    def __init__(self, matrix):
        if not is_unitary(matrix):
            raise ValueError('Quantum operations must be unitary')
        self.matrix = matrix

    def apply(self, state):
        new_vector = np.matmul(self.matrix, state.vector)
        return QuantumState(new_vector)

    def compose(self, operation):
        new_matrix = np.kron(self.matrix, operation.matrix)
        return QuantumOperation(new_matrix)

    def __repr__(self):
        return '<QuantumOperation: {}>'.format(str(self.matrix))


from math import sqrt, cos, sin, pi

# The unitary matrices of Alice and Bob's possible operations.
U_X = [[0, 1],
       [1, 0]]  # identity I

U_H = [[1 / sqrt(2), 1 / sqrt(2)],
       [1 / sqrt(2), -1 / sqrt(2)]]

U_alice_0 = [[1, 0],
             [0, 1]]  # identity I

U_alice_1 = [[cos(pi / 4), sin(pi / 4)],
             [-sin(pi / 4), cos(pi / 4)]]
U_bob_0 = [[cos(pi / 8), sin(pi / 8)],
           [-sin(pi / 8), cos(pi / 8)]]
U_bob_1 = [[cos(3 * pi / 8), sin(3 * pi / 8)],
           [-sin(3 * pi / 8), cos(3 * pi / 8)]]

U_alice_1 = RYGate(-67.5 * np.pi / 180).to_matrix()
U_bob_0 = [[1, 0],
           [0, 1]]
U_bob_1 = [[1, 0],
           [0, 1]]


# Alice and Bob win when their input (a, b)
# and their response (s, t) satisfy this relationship.
def win(a, b, s, t):
    return (a and b) == (s != t)


wins = 0

# generate "questions" in equal number
a = []
b = []
for x in range(2):
    for y in range(2):
        a.append(x)
        b.append(y)

# random.shuffle(a)
# random.shuffle(b)
state = [1 / sqrt(2), 0, 0, 1 / sqrt(2)]
# play game

result = []
for i in range(4):
    # Alice and Bob share an entangled state
    state = QuantumState([1 / sqrt(2), 0, 0, 1 / sqrt(2)])

    # The input to alice and bob is random
    # Alice chooses her operation based on her input
    if a[i] == 0:
        alice_op = QuantumOperation(U_alice_0)
    if a[i] == 1:
        alice_op = QuantumOperation(U_alice_1)

    # Bob chooses his operation based on his input
    if b[i] == 0:
        bob_op = QuantumOperation(U_bob_0)
    if b[i] == 1:
        bob_op = QuantumOperation(U_bob_1)

    # We combine Alice and Bob's operations
    combined_operation = alice_op.compose(bob_op)

    # Alice and Bob make their measurements
    new_state = combined_operation.apply(state)
    # print(new_state)
    result.append(combined_operation.apply(state).measure_analytic())

win_rate = 0
for mat in result[:-1]:
    print(mat)
    win_rate += 1 / 4 * (mat[0] + mat[3])

win_rate += 1 / 4 * (result[-1][1] + result[-1][2])
print(win_rate)
evaluation_tactic = [[1, 0, 0, 1],
          [1, 0, 0, 1],
          [1, 0, 0, 1],
          [0, 1, 1, 0]]
win_rate1 = 0
for x, riadok in enumerate(evaluation_tactic):
    for y, stlpec in enumerate(riadok):
        win_rate1 += (stlpec * result[x][y])
win_rate1 = win_rate1 * 1 / 4

# test evaluation_tactic DONE
print(win_rate1)

# assert (win_rate==win_rate1) # test evaluation_tactic DONE

#     # Convert the 4 state measurement result to two 1-bit results
#     if result == 0:
#         s, t = False, False
#     if result == 1:
#         s, t = False, True
#     if result == 2:
#         s, t = True, False
#     if result == 3:
#         s, t = True, True

#     # Check if they won and add it to the total
#     wins += win(a[i], b[i], s, t)

# print('They won this many times:', wins)
