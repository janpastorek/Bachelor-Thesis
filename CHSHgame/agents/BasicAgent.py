import numpy as np
import random

class BasicAgent:
    """ Reinforcement learning agent """

    def __init__(self, state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum, ALL_POSSIBLE_ACTIONS,
                 model_type):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.epsilon = eps  # exploration rate
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.alpha = alpha
        self.momentum = momentum
        self.model = model_type(state_size, action_size)
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

        # Run one training step using SGD - stochastic gradient descend
        self.model.sgd(state, target_full, self.alpha, self.momentum)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """ loads weights into model """
        self.model.load_weights(name)

    def save(self, name):
        """ saves weight into model """
        self.model.save_weights(name)