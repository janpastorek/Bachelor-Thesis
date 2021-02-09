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

# def anneal(self, steps=100, t_start=2, t_end=0.001):
#     # A function that finds the maximal value of the fitness function by
#     # executing the simulated annealing algorithm.
#     # Returns a state (e.g. x) for which fitness(x) is maximal.
#     ### YOUR CODE GOES HERE ###
#     x = self.random_state()
#     t = t_start
#     for i in range(steps):
#       neighbor = random.choice(self.neighbors(x))
#       ΔE = self.fitness(neighbor) - self.fitness(x)
#       if ΔE > 0: #//neighbor is better then x
#         x = neighbor
#       elif random.random() < np.math.e**(ΔE / t):          #//neighbor is worse then x
#             x = neighbor
#       t = t_start * ( t_end / t_start) ** (i/steps)
#     return x
#
# def fitness(self, x):
#     ...
#
# def neighbors(self, x, span=30, delta=0.1):
#     res = []
#     if x > -span + 3 * delta: res += [x - i * delta for i in range(1, 4)]
#     if x < span - 3 * delta: res += [x + i * delta for i in range(1, 4)]
#     return res
#
# def random_state(self):
#     return random.randrange(-180,180,0.1)

class Game:

    def __init__(self, scaler):
        self.scaler = scaler

    def play_one_episode(self, agent, env, is_train):
        # returns a list of states and corresponding returns
        # in this version we will NOT use "exploring starts" method
        # instead we will explore using an epsilon-soft policy
        state = env.reset()
        state = self.scaler.transform([state])
        done = False

        # be aware of the timing
        # each triple is s(t), a(t), r(t)
        # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)

        rew_accum = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action[0])
            next_state = self.scaler.transform([np.round(next_state, 3)])
            if is_train == 'train':
                agent.train(np.round(state, 3), action[1], reward, next_state, done)
            state = next_state.copy()
            rew_accum += reward
        # print(env.history_actions)
        return env.accuracy, rew_accum

    def evaluate_train(self, N, agent, env):
        co = "train"

        portfolio_value = []
        rewards = []

        for e in range(N):
            val, rew = self.play_one_episode(agent, env, co)
            print('episode:', end=' ')
            print(e, end=' ')
            print('acc:', end=' ')
            print(val)
            print('rew:', end=' ')
            print(rew)

            portfolio_value.append(val)  # append episode end portfolio value
            rewards.append(rew)

        # save the weights when we are done
        if co == 'train':
            # # save the DQN
            agent.save(f'linear.npz')

            # save the scaler
            with open(f'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

        return portfolio_value, rewards

    def evaluate_test(self, agent, n_questions, tactic, max_gates, env):
        co = "test"

        portfolio_value = []
        if co == 'test':
            N = 1

            # then load the previous scaler
            with open(f'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            # make sure epsilon is not 1!
            # no need to run multiple episodes if epsilon = 0, it's deterministic
            agent.epsilon = 0

            # load trained weights
            agent.load(f'linear.npz')

        # play the game num_episodes times

        for e in range(N):
            val = self.play_one_episode(agent, env, co)
            print('Test value:', end=' ')
            print(val)

            portfolio_value.append(val)  # append episode end portfolio value

        return portfolio_value