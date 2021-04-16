import itertools

import NonLocalGame


class Environment(NonLocalGame.abstractEnvironment):
    """ creates CHSH for classic deterministic strategies"""

    def __init__(self, game_type, num_players=2, n_questions=2):
        self.num_players = num_players
        self.n_questions = n_questions
        self.questions = list(itertools.product(list(range(2)), repeat=self.num_players * self.n_questions // 2))

        self.n_games = 1
        self.n_qubits = 0

        self.game_type = game_type

        self.possible_answers = dict()
        self.possible_answers[0] = (0, 1)
        self.possible_answers[1] = (0, 1)

        self.responses = list(
            itertools.product(list(range(2)),
                              repeat=self.num_players * self.n_questions //2))

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
            if list(r) == response:
                return counter
            counter += 1
        return

    def evaluate(self, question, response):
        """ :returns winning accuracy to input question based on response """
        self.state = [0 for _ in range(len(self.game_type))]
        # answer = (self.possible_answers[question[0]][response[0]], self.possible_answers[question[1]][response[1]])

        answer = [self.possible_answers[question[i]][response[i]] for i in range(len(response))]
        self.state[self.index(answer)] = 1
        return self.measure_analytic()

    def play_all_strategies(self):
        """ plays 16 different strategies,evaluate each and :returns: the best accuracy from all strategies """
        accuracies = []
        result = []


        response_list = self.response_rek(self.n_questions)
        for r in  self.response_rek(self.n_questions, response_list):
            for x, question in enumerate(self.questions):
                response_to_this_question = [response_list.__next__()[question[i]] for i in range(self.n_questions)]
                result.append(self.evaluate(question, response_to_this_question))
            accuracies.append(self.calc_accuracy(result))
            result = []

        return max(accuracies), min(accuracies)

    def response_rek(self, n):
        if (n == 0): pass
        else:
            for r in self.responses:
                yield r
                self.response_rek(n - 1)



def rule(a, b, x, y):
    return (a != b) == (x and y)


def create(game_type):
    game = [[0 for _ in range(len(game_type)) for __ in range(len(game_type))] for ___ in range(len(game_type)) for ____ in range(len(game_type))]
    for y1, riadok1 in enumerate(game_type):
        for x1, cell1 in enumerate(riadok1):
            for y2, riadok2 in enumerate(game_type):
            # for x1, cell1 in enumerate(riadok1):
                for x2, cell2 in enumerate(riadok2):
                    if (cell1 == cell2 and cell1 == 1): game[y1 * y2][x1 * x2] = 1  # TODO: ma tu byt cell1 == 1?
    return game


if __name__ == '__main__':
    game_type = [[1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [0, 1, 1, 0]]
    env = Environment(game_type, 2, 2)
    print(env.play_all_strategies())

    env = Environment(create(game_type), 2, 4)

    print(env.questions)

    print(env.play_all_strategies())
