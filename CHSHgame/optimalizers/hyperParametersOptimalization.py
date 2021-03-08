import math
import random

import matplotlib.pyplot as plt

from CHSH import get_scaler, override, Game
from GeneticAlg import GeneticAlg


class HyperParamCHSHOptimizer(GeneticAlg):

    def __init__(self, population_size=15, n_crossover=3, mutation_prob=0.05, game_type=None, CHSH=None,
                 max_gates=10, n_questions=4):
        # Initialize the population - create population of 'size' individuals,
        # each individual is a bit string of length 'word_len'.
        self.population_size = population_size
        self.n_crossover = n_crossover
        self.mutation_prob = mutation_prob
        self.population = [self.generate_individual() for _ in range(self.population_size)]
        self.for_plot = []

        self.game_type = game_type
        self.CHSH = CHSH
        self.max_gates = max_gates
        self.n_questions = n_questions

    @override
    def generate_individual(self):
        # Generate random individual.
        # To be implemented in subclasses
        # tieto hyperparametre treba optimalizovat
        GAMMA = [1, 0.9, 0.5, 0]
        MOMENTUM = [0.9, 0.85, 0.5]
        ALPHA = [1, 0.1, 0.01, 0.001]
        EPS = [1]
        EPS_DECAY = [0.995, 0.9995]
        EPS_MIN = [0.001, 0.025]
        N_EPISODES = [1000, 2000, 4000]

        return [random.choice(GAMMA), random.choice(EPS), random.choice(EPS_MIN), random.choice(EPS_DECAY),
                random.choice(MOMENTUM), random.choice(ALPHA), random.choice(N_EPISODES)]

    @override
    def fitness(self, x):
        # Returns fitness of a given individual.
        # To be implemented in subclasses
        N = math.floor(x[-1])

        env = self.CHSH.Environment(self.n_questions, self.game_type, self.max_gates)
        agent = self.CHSH.BasicAgent(len(env.state), len(self.CHSH.ALL_POSSIBLE_ACTIONS), gamma=x[0], eps=x[1], eps_min=x[2],
                                     eps_decay=x[3], alpha=x[4], momentum=x[5])
        scaler = get_scaler(env, N)
        batch_size = 32

        game = Game(scaler)
        game.evaluate_train(N, agent, env)

        fitness_individual = game.evaluate_test(agent, env)
        return fitness_individual

    @override
    def mutation(self, x, prob):
        return self.number_mutation(x, prob)

    @override
    def solve(self, max_generations, goal_fitness=1):
        best = super().solve(max_generations, goal_fitness)
        return best  # best


if __name__ == "__main__":
    ## Solve to find optimal individual
    from CHSHv02qDiscreteStatesActions import Environment

    evaluation_tactic = [[1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0]]

    ga = HyperParamCHSHOptimizer(population_size=6, n_crossover=5, mutation_prob=0.01,
                                 game_type=evaluation_tactic, CHSH=Environment)
    best = ga.solve(5)  # you can also play with max. generations
    ga.show_individual(best)

    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.axhline(y=0.853, color='r', linestyle='-')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Win rate')

    plt.plot(ga.for_plot)
