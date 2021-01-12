import qiskit as q
from qiskit.extensions import RYGate
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import tensorflow as tf
import abc

import numpy as np
from math import sqrt, cos, sin, pi
import random


class Environment(py_environment.PyEnvironment):

    def __init__(self, n_questions, tactic, max_gates, ALL_POSSIBLE_ACTIONS, discount):
        self.ALL_POSSIBLE_ACTIONS = ALL_POSSIBLE_ACTIONS
        self.discount = discount
        self.initial_state = np.array([0, (np.array([1],dtype=np.longdouble) / sqrt(2)).item(), -(np.array([1],dtype=np.float64) / sqrt(2)).item(), 0], dtype=np.float64)
        self.state = self.initial_state.copy()
        self.num_players = 2
        self.repr_state = np.array([x for n in range(self.num_players ** 2) for x in self.state], dtype=np.longdouble)

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0,
                                                        maximum=len(ALL_POSSIBLE_ACTIONS)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=self.repr_state.shape, dtype=np.float64, name='observation')
        self.pointer = 0  # time
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = []
        self.max_gates = max_gates
        self.tactic = tactic

        self.accuracy = 0.25
        self.rew_hist = [0]

        # input, generate "questions" in equal number
        self.a = []
        self.b = []
        for x in range(2):
            for y in range(2):
                self.a.append(x)
                self.b.append(y)

    def _reset(self):
        self.pointer = 0
        self.counter = 1
        self.history_actions = []
        self.accuracy = 0.25
        self.state = self.initial_state.copy()
        self.repr_state = np.array([x for n in range(self.num_players ** 2) for x in self.state], dtype=np.float64)
        return ts.restart(self.repr_state)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # Returns probabilities of 00,01,10,10 happening in matrix
    def measure_analytic(self):
        weights = [abs(a) ** 2 for a in self.state]
        return weights

    # Calculates winning accuracy / win rate based on winning tactic
    def calc_accuracy(self, tactic, result):
        win_rate = 0
        for x, riadok in enumerate(tactic):
            for y, stlpec in enumerate(riadok):
                win_rate += (stlpec * result[x][y])
        win_rate = win_rate * 1 / 4
        return win_rate

    def _step(self, action):

        # play game
        result = []
        self.history_actions.append(action)

        for g in range(self.n_questions):
            # Alice - a and Bob - b share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input, Bob too - eg. a0 if alice gets 0 as input

            self.state = self.initial_state.copy()

            for a in self.history_actions:
                action = self.ALL_POSSIBLE_ACTIONS[a]
                gate = np.array([action[3:]],dtype=np.float64)

                if self.a[g] == 0 and action[0:2] == 'a0':
                    self.state[:4] = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                               self.state[:4])

                if self.a[g] == 1 and action[0:2] == 'a1':
                    self.state[:4] = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                               self.state[:4])

                if self.b[g] == 0 and action[0:2] == 'b0':
                    self.state[:4] = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                               self.state[:4])

                if self.b[g] == 1 and action[0:2] == 'b1':
                    self.state[:4] = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                               self.state[:4])

            # norm = np.linalg.norm(self.state)
            # s = self.state / norm
            # self.state = self.state / norm
            # self.state = np.round(self.state, 10)
            self.repr_state[g * self.num_players ** 2:(g + 1) * self.num_players ** 2] = self.state
            result.append(self.measure_analytic())

        # accuracy of winning CHSH game
        before = self.accuracy
        win_rate = self.calc_accuracy(self.tactic,result)
        self.accuracy = win_rate

        # reward is the increase in accuracy
        rozdiel_acc = before - self.accuracy
        reward = 100 * (rozdiel_acc)

        print("acc: ", end="")
        print(self.accuracy)

        print("rew: ", end="")
        print(reward)

        # skonci, ak uz ma maximalny pocet bran alebo presiahol pozadovanu uroven self.accuracy
        if self.accuracy >= 0.83:
            self.rew_hist.append(self.accuracy)
            reward += 1000 * (1 / (len(self.history_actions) + 1))
            return ts.termination(self.repr_state, reward)

        # self.rew_hist.append(self.accuracy)

        return ts.transition(self.repr_state, reward, self.discount)


import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pickle

# setting environment
ACTIONS2 = ['r' + str(180 / 16 * i) for i in range(0, 8)]
ACTIONS = ['r' + str(- 180 / 16 * i) for i in range(1, 8)]
ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
PERSON = ['a', 'b']
QUESTION = ['0', '1']
ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS]  # place one gate at some place
N = 1000
num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 1  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
n_questions = 4
tactic = [[1, 0, 0, 1],
          [1, 0, 0, 1],
          [1, 0, 0, 1],
          [0, 1, 1, 0]]
max_gates = 6
discount = 0.9
eps = 0.9
env = Environment(n_questions, tactic, max_gates, ALL_POSSIBLE_ACTIONS, discount)
tf_env = tf_py_environment.TFPyEnvironment(env)

# setting agent
fc_layer_params = (100,)

q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=(len(env.repr_state),len(ALL_POSSIBLE_ACTIONS)))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    epsilon_greedy=eps,
    gamma=discount)

agent.initialize()

# setting policy
eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),
                                                tf_env.action_spec())


# time_step = tf_env.reset()
# random_policy.action(time_step)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# try compute avg return for 100 episodes
print(compute_avg_return(tf_env, random_policy, num_eval_episodes))

# setting replay buffer - storage of collected data
# batch_size = 32
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=batch_size,
    max_length=replay_buffer_max_length)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


# try collect data in 100 episodes
collect_data(tf_env, random_policy, replay_buffer, num_eval_episodes)

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
iterator = iter(dataset)

### Training Agent

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(tf_env, agent.policy, N)
returns = [avg_return]

for _ in range(num_iterations):
    print("episode: ", _)
    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_data(tf_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

# plot results
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
