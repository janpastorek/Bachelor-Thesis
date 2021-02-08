import random
from math import sqrt

import numpy as np
from sklearn.preprocessing import StandardScaler
from CHSHv05onlyGenetic import GenAlgProblem
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold

def get_scaler(env, N):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here

    states = []
    for _ in range(N):
        action = np.random.choice(ALL_POSSIBLE_ACTIONS)
        state, reward, done = env.step(action)
        states.append(np.round(state, 3))

        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


class LinearModel:
    """ A linear regression model """

    def __init__(self, input_dim, n_action, alpha, momentum):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []


    def predict(self, X):
        # make sure X is N x D
        assert (len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        # make sure X is N x D
        assert (len(X.shape) == 2)

        # the loss values are 2-D
        # normally we would divide by N only
        # but now we divide by N x K
        num_values = np.prod(Y.shape)

        # do one step of gradient descent
        # we multiply by 2 to get the exact gradient
        # (not adjusting the learning rate)
        # i.e. d/dx (x^2) --> 2x
        Yhat = self.predict(X)

        # print([X.shape, Y.shape])
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y) ** 2)
        self.losses.append(mse)

    def getLoss(self):
        return self.network.loss

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)

class Agent:
    def __init__(self, state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.epsilon = eps  # exploration rate
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.alpha = alpha
        self.momentum = momentum
        self.model = LinearModel(state_size, action_size, self.alpha, self.momentum)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            choice = random.randint(0, self.action_size - 1)
            return ALL_POSSIBLE_ACTIONS[choice], choice
        act_values = self.model.predict(state)
        choice = np.argmax(act_values[0])
        return ALL_POSSIBLE_ACTIONS[choice], choice

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class Environment:

    def __init__(self, n_questions, tactic, max_gates, num_players=2):
        self.pointer = 0  # time
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = []
        self.max_gates = max_gates
        self.tactic = tactic
        self.initial_state = np.array([0, 1 / sqrt(2), 1 / sqrt(2), 0],
                                      dtype=np.longdouble)  ## FIX ME SCALABILITY, TO PARAM
        self.state = self.initial_state.copy()
        self.num_players = num_players
        self.repr_state = np.array([x for n in range(self.num_players ** 2) for x in self.state], dtype=np.longdouble)
        self.accuracy = self.calc_accuracy([self.measure_analytic() for i in range(n_questions)])
        self.max_acc = self.accuracy
        self.min_gates = max_gates

        self.optimizer = GenAlgProblem(population_size=15, n_crossover=len(self.history_actions) - 1,
                                       mutation_prob=0.10, state=self.initial_state,
                                       history_actions=self.history_actions, tactic=self.tactic,
                                       num_players=self.num_players)
        self.visited = dict()

    def reset(self):
        self.counter = 1
        self.history_actions = []
        self.state = self.initial_state.copy()  ########## INITIAL STATE
        self.accuracy = self.calc_accuracy([self.measure_analytic() for i in range(n_questions)])
        self.repr_state = np.array([x for n in range(self.num_players ** 2) for x in self.state], dtype=np.longdouble)
        return self.repr_state

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
        win_rate = win_rate * 1 / len(tactic)
        return win_rate

    def calculateNewStateAccuracy(self, action):
        self.history_actions.append(action)
        try:
            actions, accuracy, self.repr_state = self.visited[tuple(self.history_actions)]
        except KeyError:
            self.optimizer.reInitialize(self.history_actions, len(self.history_actions) - 1)
            actions, accuracy, self.repr_state = self.optimizer.solve(22)
            self.visited[tuple(self.history_actions)] = actions, accuracy, self.repr_state
        return accuracy

    def step(self, action):

        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        # accuracy of winning CHSH game
        # reward is the increase in accuracy
        accuracyBefore = self.accuracy
        self.accuracy = self.calculateNewStateAccuracy(action)
        reward, done = self.rewardOnlyBest(accuracyBefore, done)

        # print("acc: ", end="")
        # print(self.accuracy)
        #
        # print("rew: ", end="")
        # print(reward)

        if done == True:
            print(self.visited[tuple(self.history_actions)][0])
        else:
            self.counter += 1
        return self.repr_state, reward, done

    def countGates(self):
        count = 0
        for action in self.history_actions:
            if action != "xxr0":
                count += 1
        return count

    def rewardOnlyBest(self, accuracyBefore, done):
        # reward = self.accuracy - accuracyBefore
        reward = 0

        # always award only the best (who is best changes through evolution)
        if np.round(self.accuracy, 2) > np.round(self.max_acc, 2):
            self.min_gates = len(self.history_actions)
            self.max_acc = self.accuracy
        elif np.round(self.accuracy, 2) == np.round(self.max_acc, 2):
            if self.min_gates > len(self.history_actions):
                self.min_gates = len(self.history_actions)
            self.max_acc = self.accuracy

        # skonci, ak uz ma maximalny pocet bran alebo pouzil "ukoncovaciu branu"
        if self.counter == self.max_gates or self.history_actions[-1] == "xxr0":
            done = True
            if np.round(self.max_acc, 2) == np.round(self.accuracy, 2) and self.min_gates == self.countGates():
                reward = 100 * (1 / (self.countGates() + 1)) * self.accuracy
            # elif np.round(self.max_acc, 2) == np.round(self.accuracy, 2):
            #     reward -= 1000 * (self.countGates() + 1) / self.accuracy
            else:
                reward = 0
                # reward -= 10000 * (self.countGates() + 1) / self.accuracy  # alebo tu dam tiez nejaky vzorcek

        return reward, done

    def rewardPositiveDifference(self, accuracyBefore, done):
        reward = self.accuracy - accuracyBefore
        if self.counter == self.max_gates or self.history_actions[-1] == "xxr0":
            done = True
        return reward, done




import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pickle


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


if __name__ == '__main__':
    ACTIONS = ['r0']  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in
                            ACTIONS]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append("xxr0")

    N = 4000
    n_questions = 4
    tactic = [[1, 0, 0, 1],
              [1, 0, 0, 1],
              [1, 0, 0, 1],
              [0, 1, 1, 0]]
    max_gates = 10

    env = Environment(n_questions, tactic, max_gates)

    # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
    agent = Agent(len(env.repr_state), len(ALL_POSSIBLE_ACTIONS), 0.0, 1, 0.01, 0.995, 1, 0.5)
    scaler = get_scaler(env, N)
    batch_size = 128

    # store the final value of the portfolio (end of episode)
    game = Game(scaler)
    portfolio_value, rewards = game.evaluate_train(N, agent, env)

    # plot relevant information
    # reward
    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.xlabel('Epochs')
    plt.ylabel('Reward')

    plt.plot(rewards)
    plt.show()

    # agent loss function

    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.plot(agent.model.getLoss())
    plt.show()

    # win rate
    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.axhline(y=0.853, color='r', linestyle='-')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Win rate')
    plt.plot(portfolio_value)
    plt.show()

    # save portfolio value for each episode
    np.save(f'train.npy', portfolio_value)
    portfolio_value = game.evaluate_test(agent, n_questions, tactic, max_gates, env)
    print(portfolio_value)
    a = np.load(f'train.npy')
    print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")
