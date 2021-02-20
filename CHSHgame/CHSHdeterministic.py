import itertools

import CHSH


class Environment(CHSH.abstractEnvironment):
    """ creates CHSH for classic deterministic strategies"""

    def __init__(self, evaluation_tactic):
        self.a = []
        self.b = []
        for x in range(2):
            for y in range(2):
                self.a.append(x)
                self.b.append(y)

        self.evaluation_tactic = evaluation_tactic
        self.possible_answers = dict()
        self.possible_answers[0] = (0, 1)
        self.possible_answers[1] = (0, 1)
        self.responses = list(itertools.product([0, 1], repeat=2))

    @CHSH.override
    def reset(self):
        return

    @CHSH.override
    def step(self, action):
        return

    def index(self, response):
        """ :returns index of response so that it can be mapped to state"""
        counter = 0
        for r in self.responses:
            if r == response:
                break
            counter += 1
        return counter

    def evaluate(self, question, response):
        """ :returns winning accuracy to input question based on response """
        self.state = [0 for i in range(len(self.evaluation_tactic))]
        answer = (self.possible_answers[question[0]][response[0]],self.possible_answers[question[1]][response[1]])
        self.state[self.index(answer)] = 1
        return self.measure_analytic()

    def play_all_strategies(self):
        """ plays 16 different strategies,evaluate each and :returns: the best accuracy from all strategies """
        accuracies = []
        result = []
        for a in range(len(self.possible_answers)):
            for b in range(len(self.possible_answers)):
                for q in range(len(self.evaluation_tactic)):
                    question = [self.a[q], self.b[q]]
                    result.append(self.evaluate(question, (a, b)))
                accuracies.append(self.calc_accuracy(result))
                result = []

        print(accuracies)
        return max(accuracies)
