from sklearn.preprocessing import StandardScaler


def get_scaler(env, N, ALL_POSSIBLE_ACTIONS, roundTo=2):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here
    states = []
    for _ in range(N):
        action = np.random.choice(ALL_POSSIBLE_ACTIONS)
        state, reward, done = env.step(action)
        states.append(np.round(state, roundTo))

        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def override(f):
    return f


import abc


class abstractEnvironment:

    @abc.abstractmethod
    def reset(self):
        """Return initial_time_step."""

    @abc.abstractmethod
    def step(self, action):
        """Apply action and return new time_step."""

    # Returns probabilities of 00,01,10,10 happening in matrix
    def measure_analytic(self):
        weights = [abs(a) ** 2 for a in self.state]
        return weights

    # Calculates winning accuracy / win rate based on winning tactic
    def calc_accuracy(self, result):
        win_rate = 0
        for x, riadok in enumerate(self.tactic):
            for y, stlpec in enumerate(riadok):
                win_rate += (stlpec * result[x][y])
        win_rate = win_rate * 1 / len(self.tactic)
        return win_rate

    def countGates(self):
        count = 0
        for action in self.history_actions:
            if action != "xxr0":
                count += 1
        return count


import random
from LinearModel import LinearModel


class Agent:
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
        if np.random.rand() <= self.epsilon:
            choice = random.randint(0, self.action_size - 1)
            return self.ALL_POSSIBLE_ACTIONS[choice], choice
        act_values = self.model.predict(state)
        choice = np.argmax(act_values[0])
        return self.ALL_POSSIBLE_ACTIONS[choice], choice

    def train(self, state, action, reward, next_state, done):
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
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


import warnings

warnings.filterwarnings('ignore')
import pickle
import numpy as np


class Game:

    def __init__(self, scaler, roundTo=2):
        self.scaler = scaler
        self.roundTo = roundTo

    def play_one_episode(self, agent, env, DO):
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
            next_state = self.scaler.transform([np.round(next_state, self.roundTo)])
            if DO == 'train':
                agent.train(np.round(state, self.roundTo), action[1], reward, next_state, done)
            state = next_state.copy()
            rew_accum += reward
        print(env.history_actions)
        return env.accuracy, rew_accum

    def evaluate_train(self, N, agent, env):
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
            agent.save(f'linear.npz')

            # save the scaler
            with open(f'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

        return portfolio_value, rewards

    def evaluate_test(self, agent, n_questions, tactic, max_gates, env):
        DO = "test"

        portfolio_value = []
        if DO == 'test':
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
            env.reset()
            val = self.play_one_episode(agent, env, DO)
            print('Test value:', end=' ')
            print(val)

            portfolio_value.append(val)  # append episode end portfolio value

        return portfolio_value
