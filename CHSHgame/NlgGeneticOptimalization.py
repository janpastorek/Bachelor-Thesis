import itertools
import random
from math import sqrt, pi

import matplotlib.pyplot as plt
import numpy as np
from qiskit.extensions import IGate

from NonLocalGame import abstractEnvironment, override
from optimalizers.GeneticAlg import GeneticAlg


class CHSHgeneticOptimizer(GeneticAlg, abstractEnvironment):
    """ Creates CHSH genetic optimizer """

    @override
    def __init__(self, population_size=15, n_crossover=3, mutation_prob=0.05,
                 state=[0, float(1 / sqrt(2)), -float(1 / sqrt(2)), 0],
                 history_actions=['a0r0', 'b0r0', 'a1r0', 'b1r0'], game_type=[], num_players=2, n_questions=2, best_or_worst="best"):
        # Initialize the population - create population of 'size' individuals,
        # each individual is a bit string of length 'word_len'.
        super().__init__()
        self.n_questions = n_questions
        self.best_or_worst = best_or_worst
        self.population_size = population_size
        self.n_crossover = n_crossover
        self.mutation_prob = mutation_prob
        self.num_players = num_players
        self.initial = state
        self.game_type = game_type

        self.reset(history_actions, n_crossover)

        # generate "questions" in equal number
        self.questions = list(itertools.product(list(range(self.n_questions)), repeat=self.num_players))

    @override
    def reset(self, history_actions, n_crossover):
        """ Initializes number of crossovers and CHSH environment with :param history_actions - new previous actions"""
        self.state = self.initial.copy()
        self.n_crossover = n_crossover
        self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state], dtype=np.complex128)
        self.history_actions = history_actions
        self.for_plot = []
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    @override
    def step(self, action):
        pass

    @override
    def generate_individual(self):
        """Generate random individual."""
        # tieto hyperparametre treba optimalizovat - brany
        return [str(action[0:4]) + str(random.uniform(-180, 180)) if action != 'xxr0' else 'xxr0' for action in
                self.history_actions]

    @override
    def fitness(self, x):
        """ Returns fitness of a given individual."""
        result = []

        for g, q in enumerate(self.questions):
            # Alice - a and Bob - b share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input, Bob too - eg. a0 if alice gets 0 as input
            self.state = self.initial.copy()
            self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state], dtype=np.complex128)

            for action in x:
                gate = self.get_gate(action)
                if gate == IGate: continue
                to_whom = action[0:2]
                try: gate_angle = np.array([action[4:]], dtype=np.float64)
                except ValueError: gate_angle = 0

                operation = []

                if (q[0] == 0 and to_whom == 'a0') or (q[0] == 1 and to_whom == 'a1'):
                    calc_operation = np.kron(np.identity(2), gate((gate_angle * pi / 180).item()).to_matrix())
                    operation = calc_operation
                if (q[1] == 0 and to_whom == 'b0') or (q[1] == 1 and to_whom == 'b1'):
                    calc_operation = np.kron(gate((gate_angle * pi / 180).item()).to_matrix(), np.identity(2))
                    operation = calc_operation

                if len(operation) != 0:
                    self.state = np.matmul(operation, self.state)

            self.repr_state[g * self.num_players ** 2:(g + 1) * self.num_players ** 2] = self.state.copy()

            result.append(self.measure_analytic())
        fitness_individual = self.calc_accuracy(result)
        return fitness_individual

    @override
    def number_mutation(self, x, prob):
        """ Elements of x are real numbers [0.0 .. 1.0]. Mutate (i.e. add/substract random number)
         each number in x with given probabipity."""
        potomok = x
        for poc in range(len(potomok)):
            if random.random() <= prob:
                spocitaj = [float(gate[4:]) for gate in potomok]
                priemer = sum(spocitaj) / len(spocitaj)
                sigma_na_druhu = 0

                for i in spocitaj:
                    sigma_na_druhu += (i - priemer) ** 2

                sigma_na_druhu = sigma_na_druhu / (len(spocitaj)) / 360  # Normal distribution

                if random.random() > 0.5:
                    if potomok[poc] != 'xxr0':
                        nahodne = random.uniform(0, sigma_na_druhu)
                        potomok[poc] = potomok[poc][:4] + str(float(potomok[poc][4:]) - nahodne)

                else:
                    if potomok[poc] != 'xxr0':
                        nahodne = random.uniform(0, sigma_na_druhu)
                        potomok[poc] = potomok[poc][:4] + str(float(potomok[poc][4:]) + nahodne)

        return potomok

    @override
    def mutation(self, x, prob):
        return self.number_mutation(x, prob)

    @override
    def solve(self, max_generations, goal_fitness=1):
        """Implementation of genetic algorithm. Produce generations until some
        # individual`s fitness reaches goal_fitness, or you exceed total number
        # of max_generations generations. Return best found individual. """
        best = super().solve(max_generations, goal_fitness)
        accuracy = self.fitness(best)
        return best, accuracy, self.repr_state  # all is for best


if __name__ == "__main__":
    # Solve to find optimal individual
    ACTIONS2 = ['r' + axis + "0" for axis in 'xyz']
    # ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2]  # place one gate at some place
    game = [[1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]]
    ga = CHSHgeneticOptimizer(population_size=30, n_crossover=len(ALL_POSSIBLE_ACTIONS) - 1, mutation_prob=0.1, history_actions=ALL_POSSIBLE_ACTIONS,
                              game_type=game, best_or_worst="worst")
    best = ga.solve(22)  # you can also play with max. generations
    ga.show_individual(best[0])

    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.axhline(y=0.853, color='r', linestyle='-')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Win rate')

    plt.plot(ga.for_plot)
    plt.show()
