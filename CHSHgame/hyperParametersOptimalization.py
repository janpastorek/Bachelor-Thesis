import math
import random

from CHSH import get_scaler

class GenAlgProblem:

    def __init__(self, population_size=15, n_crossover=3, mutation_prob=0.05):
        # Initialize the population - create population of 'size' individuals,
        # each individual is a bit string of length 'word_len'.
        self.population_size = population_size
        self.n_crossover = n_crossover
        self.mutation_prob = mutation_prob
        self.population = [self.generate_individual() for _ in range(self.population_size)]
        self.for_plot = []

    def generate_individual(self):
        # Generate random individual.
        # To be implemented in subclasses

        # tieto hyperparametre treba optimalizovat
        GAMMA = [1,0.9, 0.5, 0]
        MOMENTUM = [0.9, 0.85, 0.5]
        ALPHA = [1,0.1, 0.01, 0.001]
        EPS = [1]
        EPS_DECAY = [0.95, 0.995, 0.99, 0.9995]
        EPS_MIN = [0.001, 0.025]
        N_EPISODES = [1000, 2000,4000]

        return [random.choice(GAMMA), random.choice(EPS), random.choice(EPS_MIN), random.choice(EPS_DECAY),
                random.choice(MOMENTUM), random.choice(ALPHA), random.choice(N_EPISODES)]

    def show_individual(self, x):
        # Show the given individual x, either to console or graphically.
        # To be implemented in subclasses
        print(x)

    def fitness(self, x):
        # Returns fitness of a given individual.
        # To be implemented in subclasses
        N = math.floor(x[-1])

        n_questions = 4
        env = Environment(n_questions)  # , max_gates trest dlzka, trest_pod brany
        agent = Agent(len(env.state), len(ALL_POSSIBLE_ACTIONS), x[0], x[1], x[2], x[3], x[4],
                      x[5])  # gamma, eps, eps_min, eps_decay, alpha, momentum
        scaler = get_scaler(env, N)
        batch_size = 32

        game = Game(scaler)
        game.evaluate_train(N, agent, env)

        fitness_individual = game.evaluate_test(agent, n_questions)
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
                spocitaj = list(potomok)
                priemer = sum(spocitaj) / len(spocitaj)
                sigma_na_druhu = 0

                for i in spocitaj:
                    sigma_na_druhu += (i - priemer) ** 2

                sigma_na_druhu = sigma_na_druhu / (len(spocitaj) - 1) / 2000  # pocitam gausovu krivku

                if random.random() > 0.5:
                    while True:
                        nahodne = random.uniform(0, sigma_na_druhu)
                        potomok[poc] -= nahodne
                        break

                else:
                    while True:
                        nahodne = random.uniform(0, sigma_na_druhu)
                        potomok[poc] += nahodne
                        break

        return potomok[:len(potomok) - 2] + [int(math.floor(potomok[-2])), int(math.floor(potomok[-1]))]

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
            print(self.population)
            sort_population = sorted(self.population, key=lambda x: self.fitness(x), reverse=True)
            najlepsi_zatial = sort_population[0]
            self.for_plot.append(najlepsi_zatial)

            # for i in sort_population:
            #     print(self.fitness(i))

            if self.fitness(sort_population[0]) == goal_fitness:
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
        najlepsi_zatial = sort_population[0]
        self.for_plot.append(najlepsi_zatial)
        return sort_population[0]  # najlepsi


if __name__ == "__main__":
    ## Solve to find optimal individual
    ga = GenAlgProblem(population_size=6, n_crossover=5, mutation_prob=0.01)
    best = ga.solve(5)  # you can also play with max. generations
    ga.show_individual(best)

    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.axhline(y=0.853, color='r', linestyle='-')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Win rate')

    plt.plot(ga.for_plot)
