import random

from abc import ABC, abstractmethod


class GeneticAlg(ABC):
    """ Abstract genetic algorithm framwork """

    @abstractmethod
    def generate_individual(self):
        """ Generate random individual.
        To be implemented in subclasses
        """
        pass

    def show_individual(self, x):
        """ Show the given individual x, either to console or graphically."""
        print(x)

    @abstractmethod
    def fitness(self, x):
        """Returns fitness of a given individual.
        To be implemented in subclasses"""
        pass

    def crossover(self, x, y, k):
        """ Take two parents (x and y) and make two children by applying k-point
        crossover. Positions for crossover are chosen randomly."""
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
        """ Elements of x are 0 or 1. Mutate (i.e. change) each element of x with given probability."""
        potomok = x
        for poc in range(len(potomok)):
            if random.random() <= prob:
                if potomok[poc] == 1:
                    potomok[poc] = 0
                else:
                    potomok[poc] = 1
        return potomok

    @abstractmethod
    def number_mutation(self, x, prob):
        """ Elements of x are real numbers [0.0 .. 1.0]. Mutate (i.e. add/substract random number)
        each number in x with given probabipity."""
        pass

    @abstractmethod
    def mutation(self, x, prob):
        """ Decides which mutation will occur. """
        pass

    @abstractmethod
    def solve(self, max_generations, goal_fitness=1):
        """ Implementation of genetic algorithm. Produce generations until some
         individual`s fitness reaches goal_fitness, or you exceed total number
         of max_generations generations. Return best found individual."""
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
        return najlepsi
