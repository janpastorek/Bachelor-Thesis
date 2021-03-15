import itertools

import NonLocalGame


class Environment(NonLocalGame.abstractEnvironment):
    """ creates CHSH for classic deterministic strategies"""

    def __init__(self, game_type, num_players=2, n_questions=2):
        self.num_players = num_players
        self.n_questions = n_questions
        self.questions = list(itertools.product(list(range(self.n_questions)), repeat=self.num_players))

        # self.a = []
        # self.b = []
        # for x in range(2):
        #     for y in range(2):
        #         self.a.append(x)
        #         self.b.append(y)

        self.game_type = game_type

        self.possible_answers = dict()
        self.possible_answers[0] = (0, 1)
        self.possible_answers[1] = (0, 1)

        self.responses = list(
            itertools.product(list(range(n_questions)),
                              repeat=self.num_players))

    @NonLocalGame.override
    def reset(self):
        return

    @NonLocalGame.override
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
        self.state = [0 for _ in range(len(self.game_type))]
        answer = (self.possible_answers[question[0]][response[0]], self.possible_answers[question[1]][response[1]])
        self.state[self.index(answer)] = 1
        return self.measure_analytic()

    def play_all_strategies(self):
        """ plays 16 different strategies,evaluate each and :returns: the best accuracy from all strategies """
        accuracies = []
        result = []
        # for a in range(len(self.possible_answers)):
        #     for b in range(len(self.possible_answers)):
        #         for q in range(len(self.game_type)):
        #             question = [self.a[q], self.b[q]]
        #             result.append(self.evaluate(question, (a, b)))
        #         accuracies.append(self.calc_accuracy(result))
        #         result = []

        # TODO: toto treba checknut ci je to ok ?! 
        for r_A in self.responses:
            for r_B in self.responses:
                for x, question in enumerate(self.questions):
                    response_to_this_question = r_A[question[0]], r_B[question[1]]
                    result.append(self.evaluate(question, response_to_this_question))
                accuracies.append(self.calc_accuracy(result))
                result = []

        # for q in range(len(self.game_type)):
        #     question = [self.a[q], self.b[q]]
        #     for a in self.possible_answers[question[0]]:
        #         self.possible_answers[question[0]] = a
        #         for b in self.possible_answers[question[1]]:
        #             self.possible_answers[question[1]] = b
        #             result.append(self.evaluate(question, (a, b)))
        #
        #     accuracies.append(self.calc_accuracy(result))
        #     result = []

        # print(accuracies)
        return max(accuracies), min(accuracies)
