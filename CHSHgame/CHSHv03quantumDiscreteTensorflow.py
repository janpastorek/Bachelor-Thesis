import qiskit as q
from qiskit.extensions import RYGate
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec, tensor_spec, BoundedTensorSpec, TensorSpec, ArraySpec, BoundedArraySpec
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
        self.initial_state = np.array([0, (np.array([1], dtype=np.float64) / sqrt(2)).item(),
                                       -(np.array([1], dtype=np.float64) / sqrt(2)).item(), 0], dtype=np.float64)
        self.state = self.initial_state.copy()
        self.num_players = 2
        self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state], dtype=np.float64)

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0,
                                                        maximum=len(ALL_POSSIBLE_ACTIONS) - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=self.repr_state.shape, dtype=np.float64,
                                                             name='observation')
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = []
        self.max_gates = max_gates
        self.tactic = tactic

        self.accuracy = self.calc_accuracy([self.measure_analytic() for _ in range(n_questions)])
        self.max_acc = self.accuracy
        self.rew_hist = [0]

        # input, generate "questions" in equal number
        self.a = []
        self.b = []
        for x in range(2):
            for y in range(2):
                self.a.append(x)
                self.b.append(y)

    def _reset(self):
        self.counter = 1
        self.history_actions = []
        self.state = self.initial_state.copy()
        self.accuracy = self.calc_accuracy([self.measure_analytic() for _ in range(n_questions)])
        self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state], dtype=np.float64)
        return ts.restart(self.repr_state)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # Returns probabilities of 00,01,10,10 happening in matrix
    def measure_analytic(self):
        weights = [abs(a) ** 2 for a in self.state]
        return weights

    # Calculates winning accuracy / win rate based on winning evaluation_tactic
    def calc_accuracy(self, result):
        win_rate = 0
        for x, riadok in enumerate(self.tactic):
            for y, stlpec in enumerate(riadok):
                win_rate += (stlpec * result[x][y])
        win_rate = win_rate * 1 / 4
        return win_rate

    def calculateState(self, history_actions):
        result = []
        for g in range(self.n_questions):
            # Alice - a and Bob - b share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input, Bob too - eg. a0 if alice gets 0 as input

            self.state = self.initial_state.copy()  ########## INITIAL STATE

            for action in history_actions:
                gate = np.array([action[3:]], dtype=np.float64)

                if self.a[g] == 0 and action[0:2] == 'a0':
                    self.state = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                           self.state)

                if self.a[g] == 1 and action[0:2] == 'a1':
                    self.state = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                           self.state)

                if self.b[g] == 0 and action[0:2] == 'b0':
                    self.state = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                           self.state)

                if self.b[g] == 1 and action[0:2] == 'b1':
                    self.state = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                           self.state)

            self.repr_state[g * self.num_players ** 2:(g + 1) * self.num_players ** 2] = self.state.copy()

            result.append(self.measure_analytic())
        return result

    def _step(self, action):
        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        # play game
        self.history_actions.append(self.ALL_POSSIBLE_ACTIONS[action])
        result = self.calculateState(self.history_actions)

        # accuracy of winning CHSH game
        before = self.accuracy
        self.accuracy = self.calc_accuracy(result)

        # reward is the increase in accuracy
        rozdiel_acc = self.accuracy - before
        reward = rozdiel_acc * 100

        # skonci, ak uz ma maximalny pocet bran
        if self.accuracy >= self.max_acc:
            self.max_acc = self.accuracy
            reward += 5 * (1 / (self.countGates() + 1))  # alebo za count_gates len(history_actuons)

        if self.counter == self.max_gates:
            done = True
            reward += 50 * (1 / (self.countGates() + 1))
            return ts.termination(np.array(np.round(self.repr_state, 3)), reward)
        if done == False:
            self.counter += 1
        return ts.transition(np.array(np.round(self.repr_state, 3)), reward=reward, discount=self.discount)

    def countGates(self):
        count = 0
        for action in self.history_actions:
            if action != "xxr0":
                count += 1
        return count


import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    # setting environment
    ACTIONS2 = ['r' + str(180 / 16 * i) for i in range(1, 9)]
    ACTIONS = ['r' + str(- 180 / 16 * i) for i in range(1, 9)]
    ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in
                            ACTIONS2]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append("xxr0")
    N = 1000
    num_iterations = 6000  # @param {type:"integer"}

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
    max_gates = 10
    discount = 0.9
    eps = 0.9
    env = Environment(n_questions, tactic, max_gates, ALL_POSSIBLE_ACTIONS, discount)
    utils.validate_py_environment(env, episodes=5)
    tf_env = tf_py_environment.TFPyEnvironment(env)

    # setting agent
    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))


    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # it's output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
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

    time_step = tf_env.reset()


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


    # # try compute avg return for 100 episodes
    # print(compute_avg_return(tf_env, random_policy, num_eval_episodes))

    # setting replay buffer - storage of collected data
    batch_size = 1
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


    collect_data(tf_env, random_policy, replay_buffer, initial_collect_steps)

    # # try collect data in 100 episodes
    # collect_data(tf_env, random_policy, replay_buffer, num_eval_episodes)

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
    avg_return = compute_avg_return(tf_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(tf_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(tf_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    # plot results
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
