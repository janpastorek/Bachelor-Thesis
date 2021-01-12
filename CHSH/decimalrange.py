import numpy as np
from math import sqrt, pi

from qiskit.circuit.library import RYGate

print(np.finfo(np.longdouble))



s = np.array([0, 1 / sqrt(2), 1/ sqrt(2), 0], dtype=np.longdouble)
s1 = s.copy()

s = np.matmul(np.kron(np.identity(2), RYGate((25 * pi / 180)).to_matrix()),
          s)

s = np.matmul(np.kron(RYGate((-25 * pi / 180)).to_matrix(),np.identity(2)),
          s)

s1 = np.matmul(np.kron(RYGate((-25 * pi / 180)).to_matrix(),np.identity(2)),
          s1)

s1 = np.matmul(np.kron(np.identity(2), RYGate((25 * pi / 180)).to_matrix()),
          s1)

print(s, s1) ## numerical instability