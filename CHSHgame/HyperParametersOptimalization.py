import math
import random

import NonLocalGame
from NonLocalGame import get_scaler, override, Game
from agents.BasicAgent import BasicAgent
from agents.DQNAgent import DQNAgent
from models.LinearModel import LinearModel
from optimalizers.GeneticAlg import GeneticAlg
from sklearn.preprocessing import OneHotEncoder


class HyperParamCHSHOptimizer(GeneticAlg):
    """ Works only for DiscreteStatesActions.Environment because of different init parameters """

    def __init__(self, population_size=15, n_crossover=3, mutation_prob=0.05, game_type=None, CHSH=None,
                 max_gates=10, n_questions=2, ALL_POSSIBLE_ACTIONS=None, agent_type=BasicAgent, best_or_worst="best"):
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
        self.ALL_POSSIBLE_ACTIONS = ALL_POSSIBLE_ACTIONS

        self.best_or_worst = best_or_worst
        self.agent_type = agent_type

    @override
    def generate_individual(self):
        # Generate random individual.
        # Parameters to be optimalized.
        GAMMA = [1, 0.9, 0.5, 0.1, 0]
        MOMENTUM = [0.9, 0.85, 0.5]
        ALPHA = [1, 0.1, 0.01, 0.001]
        EPS = [1]
        EPS_DECAY = [0.99995, 0.9995, 0.9998]
        EPS_MIN = [0.001]
        N_EPISODES = [2000, 3000, 4000]
        HIDDEN_LAYERS = [[20, 20], [20], [30, 30], [30, 30, 30]]
        BATCH_SIZE = [32, 64, 128, 256]
        reward_functions = [f for name, f in NonLocalGame.abstractEnvironment.__dict__.items()
                            if callable(f) and "reward" in name]

        return [random.choice(GAMMA), random.choice(EPS),
                random.choice(EPS_MIN), random.choice(EPS_DECAY),
                random.choice(MOMENTUM), random.choice(ALPHA),
                random.choice(N_EPISODES), random.choice(HIDDEN_LAYERS),
                random.choice(reward_functions),  random.choice(BATCH_SIZE)]

    @override
    def fitness(self, x):
        # Returns fitness of a given individual.
        # To be implemented in subclasses
        N = math.floor(x[-4])

        env = self.CHSH(self.n_questions, self.game_type, self.max_gates, reward_function=x[-2], anneal=True)

        if self.agent_type == BasicAgent:
            agent = BasicAgent(state_size=len(env.repr_state), action_size=len(self.ALL_POSSIBLE_ACTIONS), gamma=x[0], eps=x[1], eps_min=x[2],
                               eps_decay=x[3], alpha=x[4], momentum=x[5], ALL_POSSIBLE_ACTIONS=self.ALL_POSSIBLE_ACTIONS,
                               model_type=LinearModel)
            scaler = get_scaler(env, N, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS)

        else:
            # transform actions to noncorellated encoding
            encoder = OneHotEncoder(drop='first', sparse=False)
            # transform data
            onehot = encoder.fit_transform(ALL_POSSIBLE_ACTIONS)
            onehot_to_action = dict()
            action_to_onehot = dict()
            for a, a_encoded in enumerate(onehot):
                onehot_to_action[str(a_encoded)] = a
                action_to_onehot[a] = str(a_encoded)

            HIDDEN_LAYERS = x[-3]
            agent = DQNAgent(state_size=env.state_size, action_size=len(ALL_POSSIBLE_ACTIONS), gamma=x[0], eps=x[1], eps_min=x[2],
                             eps_decay=x[3], ALL_POSSIBLE_ACTIONS=self.ALL_POSSIBLE_ACTIONS, learning_rate=x[4], hidden_layers=len(HIDDEN_LAYERS),
                             hidden_dim=HIDDEN_LAYERS, onehot_to_action=onehot_to_action, action_to_onehot=action_to_onehot)
            scaler = None

        game = Game(scaler, batch_size=x[-1])
        game.evaluate_train(N, agent, env)

        fitness_individual = game.evaluate_test(agent, env)
        return fitness_individual

    @override
    def number_mutation(self, x, prob):
        """ Elements of x are real numbers [0.0 .. 1.0]. Mutate (i.e. add/substract random number)
         each number in x with given probabipity."""
        potomok = x[:-3]
        for poc in range(len(potomok)):
            if random.random() <= prob:  # posledne argumenty nebudu mutovat (N_EPISODES, REWARD_FUNCTION)
                spocitaj = list(potomok)
                priemer = sum(spocitaj) / len(spocitaj)
                sigma_na_druhu = 0

                for i in spocitaj:
                    sigma_na_druhu += (i - priemer) ** 2

                sigma_na_druhu = sigma_na_druhu / (len(spocitaj) - 1)  # pocitam gausovu krivku

                if random.random() > 0.5:
                    nahodne = random.uniform(0, sigma_na_druhu)
                    if potomok[poc] - nahodne >= 0:
                        potomok[poc] -= nahodne

                else:
                    nahodne = random.uniform(0, sigma_na_druhu)
                    potomok[poc] += nahodne

                potomok[poc] = abs(potomok[poc])

        return potomok + x[-3:]

    @override
    def mutation(self, x, prob):
        return self.number_mutation(x, prob)

    @override
    def solve(self, max_generations, goal_fitness=1):
        best = super().solve(max_generations, goal_fitness)
        return best  # best


if __name__ == "__main__":
    # Hyperparameters setting
    ACTIONS = [q + axis + "0" for axis in 'xyz' for q in 'ra']
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [[p + q + a] for p in PERSON for q in QUESTION for a in ACTIONS]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append(["xxr0"])

    ## Solve to find optimal individual
    from NlgDiscreteStatesActions import Environment

    game_type = [[1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [0, 1, 1, 0]]

    ga = HyperParamCHSHOptimizer(population_size=6, n_crossover=5, mutation_prob=0.05,
                                 game_type=game_type, CHSH=Environment, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, agent_type=DQNAgent)
    best = ga.solve(5)  # you can also play with max. generations
    ga.show_individual(best)
