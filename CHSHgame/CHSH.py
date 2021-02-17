from qiskit.extensions import RYGate, RZGate, RXGate, IGate
from sklearn.preprocessing import StandardScaler


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


def override(f):
    return f


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
        self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state], dtype=np.float64)
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
            if action != "xxr0":
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
from LinearModel import LinearModel


class Agent:
    """ Reinforcement learning agent """

    def __init__(self, state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum, ALL_POSSIBLE_ACTIONS):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.epsilon = eps  # exploration rate
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.alpha = alpha
        self.momentum = momentum
        self.model = LinearModel(state_size, action_size)
        self.ALL_POSSIBLE_ACTIONS = ALL_POSSIBLE_ACTIONS

    def act(self, state):
        """ :returns action based on neural model prediction / epsilon greedy """

        if np.random.rand() <= self.epsilon:
            choice = random.randint(0, self.action_size - 1)
            return self.ALL_POSSIBLE_ACTIONS[choice], choice
        act_values = self.model.predict(state)
        choice = np.argmax(act_values[0])
        return self.ALL_POSSIBLE_ACTIONS[choice], choice

    def train(self, state, action, reward, next_state, done):
        """ performs one training step of neural network """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step
        self.model.sgd(state, target_full, self.alpha, self.momentum)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """ loads weights into model """
        self.model.load_weights(name)

    def save(self, name):
        """ saves weight into model """
        self.model.save_weights(name)


import warnings

warnings.filterwarnings('ignore')
import pickle
import numpy as np


class Game:
    """ creates CHSH game framework for easier manipulation """

    def __init__(self, scaler, round_to=2):
        self.scaler = scaler
        self.round_to = round_to

    def play_one_episode(self, agent, env, DO):
        """ Plays one episode of CHSH training
        :returns last accuracy acquired and rewards from whole episode """
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
            next_state = self.scaler.transform([np.round(next_state, self.round_to)])
            if DO == 'train':
                agent.train(np.round(state, self.round_to), action[1], reward, next_state, done)
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
    interesting_evaluation_tactics = dict()
    for tactic in tactics:
        try:
            if interesting_evaluation_tactics[(tactic[1], tactic[0], tactic[3], tactic[2])]: pass
        except KeyError:
            try:
                if interesting_evaluation_tactics[(tactic[3], tactic[2], tactic[1], tactic[0])]: pass
            except KeyError:
                interesting_evaluation_tactics[tactic] = True

    print(len(interesting_evaluation_tactics.keys()))
    return interesting_evaluation_tactics.keys()


import CHSHdeterministic


def play_deterministic(tactic):
    """ Learns to play the best classic strategy according to tactic """
    env = CHSHdeterministic.Environment(tactic)
    best = env.play_all_strategies()
    return best


import CHSHv02quantumDiscreteStatesActions


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

    N = 6000
    n_questions = 4
    max_gates = 10
    round_to = 2
    env = CHSHv02quantumDiscreteStatesActions.Environment(n_questions, evaluation_tactic, max_gates)

    # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
    agent = Agent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=0.9, eps=1, eps_min=0.01,
                  eps_decay=0.9995, alpha=0.001, momentum=0.9, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS)
    scaler = get_scaler(env, N, ALL_POSSIBLE_ACTIONS, round_to=round_to)
    batch_size = 128

    # store the final value of the portfolio (end of episode)
    game = Game(scaler, round_to=round_to)
    game.evaluate_train(N, agent, env)
    accuracy, reward = game.evaluate_test(agent, env)
    return accuracy


def max_entangled_difference(n):
    """ Prints evaluation tactics that had the biggest difference between classical and quantum strategy """
    cutTactics = generate_only_interesting_tactics(n)

    differences = []
    for tactic in cutTactics:
        classical_max = play_deterministic(tactic)
        quantum_max = play_quantum(tactic)
        difference_win_rate = quantum_max - classical_max
        differences.append((tactic, difference_win_rate))

    differences.sort(key=lambda x: x[1])  # sorts according to difference in winning rate
    for tactic, difference_win_rate in differences:
        print("evaluation_tactic = ", tactic)
        print("difference = ", difference_win_rate)


if __name__ == '__main__':
    # max_entangled_difference(size=4)
    evaluation_tactic = [[1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0]]
    print(play_deterministic(evaluation_tactic))
