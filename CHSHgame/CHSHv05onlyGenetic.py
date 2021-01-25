import math
import random
from math import sqrt, pi

import matplotlib.pyplot as plt
import numpy as np
from qiskit.extensions import RYGate


class GenAlgProblem:

    def __init__(self, population_size=15, n_crossover=3, mutation_prob=0.05,
                 state=[0, float(1 / sqrt(2)), -float(1 / sqrt(2)), 0],
                 history_actions=['a0r0', 'b0r0', 'a1r0', 'b1r0'], tactic=[], num_players=2):
        # Initialize the population - create population of 'size' individuals,
        # each individual is a bit string of length 'word_len'.
        self.population_size = population_size
        self.n_crossover = n_crossover
        self.mutation_prob = mutation_prob
        self.history_actions = history_actions
        self.population = [self.generate_individual() for _ in range(self.population_size)]
        self.num_players = num_players
        self.state = state
        self.repr_state = np.array([x for n in range(self.num_players ** 2) for x in self.state], dtype=np.longdouble)
        self.initial = state
        self.for_plot = []
        self.tactic = tactic

        # generate "questions" in equal number
        self.a = []
        self.b = []
        for x in range(2):
            for y in range(2):
                self.a.append(x)
                self.b.append(y)

    def generate_individual(self):
        # Generate random individual.
        # To be implemented in subclasses
        # tieto hyperparametre treba optimalizovat - brany
        return [str(action[0:3]) + str(random.uniform(-180, 180)) if action != 'xxr0' else 'xxr0' for action in
                self.history_actions]

    def show_individual(self, x):
        # Show the given individual x, either to console or graphically.
        # To be implemented in subclasses
        print(x)

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

    def fitness(self, x):
        # Returns fitness of a given individual.
        # To be implemented in subclasses
        result = []
        for g in range(4):
            # Alice and Bob share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input
            self.state = self.initial.copy()  ########## INITIAL STATE
            self.repr_state = np.array([x for n in range(self.num_players ** 2) for x in self.state],
                                       dtype=np.longdouble)

            for action in x:
                gate = np.array([action[3:]], dtype=np.longdouble)

                if self.a[g] == 0 and action[0:2] == 'a0':  ## FIX ME SCALABILITY, TO PARAM
                    self.state = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                           self.state)

                if self.a[g] == 1 and action[0:2] == 'a1':  ## FIX ME SCALABILITY, TO PARAM
                    self.state = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                           self.state)

                if self.b[g] == 0 and action[0:2] == 'b0':  ## FIX ME SCALABILITY, TO PARAM
                    self.state = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                           self.state)

                if self.b[g] == 1 and action[0:2] == 'b1':  ## FIX ME SCALABILITY, TO PARAM
                    self.state = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                           self.state)

            self.repr_state[g * self.num_players ** 2:(g + 1) * self.num_players ** 2] = self.state.copy()

            result.append(self.measure_analytic())

        fitness_individual = self.calc_accuracy(result)
        return fitness_individual

    def crossover(self, x, y, k):
        # Take two parents (x and y) and make two children by applying k-point
        # crossover. Positions for crossover are chosen randomly.
        oddelovace = [0, len(x)]

        for i in range(k):
            oddelovace.append(random.choice(range(len(x))))

        oddelovace = sorted(oddelovace)

        x_new, y_new = x[:], y[:]

        for i in range(1, len(oddelovace), 2):
            terajsi = oddelovace[i]
            predosly = oddelovace[i - 1]

            if predosly != terajsi:
                x_new[predosly:terajsi], y_new[predosly:terajsi] = y[predosly:terajsi], x[predosly:terajsi]  # krizenie

        return (x_new, y_new)

    def boolean_mutation(self, x, prob):
        # Elements of x are 0 or 1. Mutate (i.e. change) each element of x with given probability.
        potomok = x
        for poc in range(len(potomok)):
            if random.random() <= prob:
                if potomok[poc] == 1:
                    potomok[poc] = 0
                else:
                    potomok[poc] = 1
        return potomok

    def number_mutation(self, x, prob):
        # Elements of x are real numbers [0.0 .. 1.0]. Mutate (i.e. add/substract random number)
        # each number in x with given probabipity.
        potomok = x
        for poc in range(len(potomok)):

            if random.random() <= prob:
                spocitaj = [float(gate[3:]) for gate in potomok]
                priemer = sum(spocitaj) / len(spocitaj)
                sigma_na_druhu = 0

                for i in spocitaj:
                    sigma_na_druhu += (i - priemer) ** 2

                sigma_na_druhu = sigma_na_druhu / (len(spocitaj)) / 360  # pocitam gausovu krivku

                if random.random() > 0.5:
                    if potomok[poc] != 'xxr0':
                        nahodne = random.uniform(0, sigma_na_druhu)
                        potomok[poc] = potomok[poc][:3] + str(float(potomok[poc][3:]) - nahodne)

                else:
                    if potomok[poc] != 'xxr0':
                        nahodne = random.uniform(0, sigma_na_druhu)
                        potomok[poc] = potomok[poc][:3] + str(float(potomok[poc][3:]) + nahodne)

        return potomok

    def mutation(self, x, prob):
        mutacia = self.number_mutation(x, prob)
        return mutacia

    def solve(self, max_generations, goal_fitness=1):
        # Implementation of genetic algorithm. Produce generations until some
        # individual`s fitness reaches goal_fitness, or you exceed total number
        # of max_generations generations. Return best found individual.
        while max_generations != 0:
            # print(max_generations)
            max_generations -= 1

            # najdem najlepsieho, ci uz nieje v cieli, a zaroven vysortujem populaciu na polku
            # print(self.population)
            sort_population = sorted(self.population, key=lambda x: self.fitness(x), reverse=True)
            najlepsi_zatial = self.fitness(sort_population[0])
            self.for_plot.append(najlepsi_zatial)

            # for i in sort_population:
            #     print(self.fitness(i))

            if najlepsi_zatial == goal_fitness:
                return sort_population[0]

            polka = len(sort_population) // 2
            self.population = sort_population[:polka]  # treba zakomentovat ak ideme pouzit tournament selection

            # BONUS tournament selection   - treba zakomentovat riadok nad tymto a odkomentovat pod tymto

            ##            novy = []
            ##            for x in range(polka):
            ##                best = None
            ##                for i in range(2): # dvaja budu stale sutazit
            ##                    ind = self.population[random.randrange(0, len(self.population))]
            ##                    if (best == None) or self.fitness(ind) > self.fitness(best):
            ##                        best = ind
            ##                novy.append(best)
            ##
            ##            self.population = novy[:]

            # mutacie a skrizenie
            deti = []
            for i in range(len(self.population)):

                x = random.choice(self.population)  # rodicia
                y = random.choice(self.population)

                dvaja_potomci = self.crossover(x, y, self.n_crossover)  # skrizenie

                for ptmk in dvaja_potomci:
                    potomok = self.mutation(ptmk, self.mutation_prob)  # mutacie
                    deti.append(potomok)

            # necham len tu najlepsiu polovicu deti
            sort_deti = sorted(deti, key=lambda x: self.fitness(x), reverse=True)

            # tu uz dotvaram novu generaciu teda polka rodicov a polka deti
            polka = len(sort_deti) // 2
            deti = sort_deti[:polka]
            for i in deti:
                self.population.append(i)  # tu uz dotvaram celkovu novu generaciu

        sort_population = sorted(self.population, key=lambda x: self.fitness(x), reverse=True)
        najlepsi = sort_population[0]
        self.for_plot.append(self.fitness(najlepsi))
        accuracy = self.fitness(najlepsi)
        return najlepsi, accuracy, self.repr_state  # najlepsi


if __name__ == "__main__":
    # Solve to find optimal individual
    history_actions = ['a0r0', 'b0r0', 'a1r0', 'b1r0']
    tactic = [[1, 0, 0, 1],
              [1, 0, 0, 1],
              [1, 0, 0, 1],
              [0, 1, 1, 0]]
    ga = GenAlgProblem(population_size=15, n_crossover=3, mutation_prob=0.05, history_actions=history_actions,
                       tactic=tactic)
    best = ga.solve(50)  # you can also play with max. generations
    ga.show_individual(best[0])

    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.axhline(y=0.853, color='r', linestyle='-')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Win rate')

    plt.plot(ga.for_plot)
    plt.show()
