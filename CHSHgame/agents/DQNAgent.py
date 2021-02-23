import numpy as np
import torch
import torch.nn as nn

from models.MLPModel import MLP

### The experience replay memory ###
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


def predict(model, np_states):
    with torch.no_grad():
        inputs = torch.from_numpy(np_states.astype(np.float32))
        output = model(inputs)
        # print("output:", output)
        return output.numpy()


def train_one_step(model, criterion, optimizer, inputs, targets):
    # convert to tensors
    inputs = torch.from_numpy(inputs.astype(np.float32))
    targets = torch.from_numpy(targets.astype(np.float32))

    # zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    loss.backward()
    optimizer.step()

class DQNAgent(object):
    def __init__(self, state_size, action_size, gamma, eps, eps_min, eps_decay, ALL_POSSIBLE_ACTIONS, learning_rate, hidden_layers, hidden_dim):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = gamma  # discount rate
        self.epsilon = eps  # exploration rate
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.model = MLP(state_size, action_size, hidden_dim=hidden_dim, n_hidden_layers=hidden_layers)

        self.ALL_POSSIBLE_ACTIONS = ALL_POSSIBLE_ACTIONS

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return np.random.choice(self.action_size)
    #     act_values = predict(self.model, state)
    #     return np.argmax(act_values[0])  # returns action

    def act(self, state):
        """ :returns action based on neural model prediction / epsilon greedy """

        if np.random.rand() <= self.epsilon:
            choice = np.random.choice(self.action_size)
            return self.ALL_POSSIBLE_ACTIONS[choice], choice
        act_values = predict(self.model,state)
        choice = np.argmax(act_values[0])
        return self.ALL_POSSIBLE_ACTIONS[choice], choice

    def replay(self, batch_size=32):
        # first check if replay buffer contains enough data
        if self.memory.size < batch_size:
            return

        # sample a batch of data from the replay memory
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']

        # Calculate the target: Q(s',a)
        target = rewards + (1 - done) * self.gamma * np.amax(predict(self.model, next_states), axis=1)

        # With the PyTorch API, it is simplest to have the target be the
        # same shape as the predictions.
        # However, we only need to update the network for the actions
        # which were actually taken.
        # We can accomplish this by setting the target to be equal to
        # the prediction for all values.
        # Then, only change the targets for the actions taken.
        # Q(s,a)
        target_full = predict(self.model, states)
        target_full[np.arange(batch_size), actions] = target

        # Run one training step
        train_one_step(self.model, self.criterion, self.optimizer, states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
