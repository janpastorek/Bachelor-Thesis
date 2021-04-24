"""
evaluating classical strategies for N parallel CHSH games
inspired by Daniel Nagaj's solution, added memoization and C libraries + encapsulation and abstracion
"""

import itertools

import numpy as np

import NonLocalGame

class Environment(NonLocalGame.abstractEnvironment):

    def __init__(self, num_players=2, n_questions=2, n_games=2):
        self.num_players = num_players
        self.n_questions = n_questions

        self.n_games =n_games
        self.n_qubits = 0

        self.questions_vectors = itertools.product(list(range(2)),  # vectors can be composed of 0s or 1s
                              repeat=(self.n_games * pow(2, self.n_games)))

        self.memoization = dict()

        self.questions = list(
            itertools.product(list(range(2)),  # vectors can be composed of 0s or 1s
                              repeat=self.n_games))

    @NonLocalGame.override
    def reset(self):
        return

    @NonLocalGame.override
    def step(self, action):
        return

    def evaluate_CHSH(self, aa,bb,xx,yy):
        # evaluates the CHSH game, questions x,y, answers a,b
        # if aa^bb == xx*yy:  return 1
        # else:     return 0
        return (aa != bb) == (xx and yy)
        #return aa^bb == xx*yy # a xor b ==? x and y

    def evaluate_parallelCHSH(self, Aanswer, Banswer, NN, printout):
        # evaluate NN parallel CHSH games
        # for answers answers Aanswer,Banswer
        overallwin = 0 # for how many question sets did AB win all the NN games in parallel?
        questionvectorAlice = [0]*NN # both this and questionvectorBob are binary strings of length NN
        #print('strategy lengths', len(Aanswer),len(Banswer))
     #   print('answer strategies', Aanswer,Banswer)
        for kk in range(0,pow(2,NN)): # go over different sets of Alice's questions
            questionvectorBob = [0]*(NN)
            for ll in range(0,pow(2,NN)): # go over different sets of Bob's questions
     #           print('questions for A:', questionvectorAlice, ', and for B:', questionvectorBob)
                wincounter = 0 # counting how many of the NN games they win for a given question set
                if printout==1: evalvector = [0]*NN
                for m in range(0,NN): # go over the NN answers
                    # the "m"-th bit of Alice's answer to question set numbered "kk" is Aanswer[kk*NN+m]

                        #win_game_mm = evaluate_CHSH(Aanswer[kk*NN+m],Banswer[ll*NN+m],questionvectorAlice[m],questionvectorBob[m])
                        #wincounter = wincounter + win_game_mm
                        #print(win_game_mm,wincounter)
                    wincounter = wincounter + self.evaluate_CHSH(Aanswer[kk*NN+m],Banswer[ll*NN+m],questionvectorAlice[m],questionvectorBob[m])
                    if printout==1: evalvector[m] = self.evaluate_CHSH(Aanswer[kk*NN+m],Banswer[ll*NN+m],questionvectorAlice[m],questionvectorBob[m])
                if wincounter == NN: # only if we won all NN parallel ones
                    overallwin = overallwin + 1

                if printout==1:
                    print('Ali questions', questionvectorAlice)
                    print('Bob questions', questionvectorBob)
                    print('Alice answers', Aanswer[kk*NN:(kk+1)*NN])
                    print('Bob   answers', Banswer[ll*NN:(ll+1)*NN])
                    print('evaluations  ', evalvector)
                    print('won ',sum(evalvector),'/',NN, ' so they ')
                    if wincounter==NN:
                        print('+ WON the composite game')
                    else:
                        print('- LOST this composite game')
                    print('overall won for ',overallwin,'/',kk*pow(2,NN)+ll+1,' possible sets of questions')
                    print('---------------------------------------')

                questionvectorBob = self.binary_add(questionvectorBob,(ll+1)%(self.n_questions*self.n_games))
            questionvectorAlice = self.binary_add(questionvectorAlice,(kk+1)%(self.n_questions*self.n_games))
            #     questionvectorBob = self.binary_add(questionvectorBob,(ll))
            # questionvectorAlice = self.binary_add(questionvectorAlice,(kk))
     #   print('winning probability ',overallwin/pow(2,2*NN))
        return overallwin/pow(2,2*NN) # return the winning probability (count question sets/number of question sets)

    def binary_add(self,blist, n):
        # expect a list of length N with values 0/1
        return self.questions[n]

    def binary_add1(self,blist, n):
        # expect a list of length N with values 0/1
        try:
            return self.memoization[n]
        except KeyError:
            res = self.questions_vectors.__next__()
            self.memoization[n] = res
            return res

        # carry = 1
        # k = 0
        # N = len(blist)
        # newblist = blist
        # while k<N:
        #     if blist[k]==0:
        #         newblist[k] = 1
        #         k = N
        #     else:
        #         newblist[k] = 0
        #         k = k+1
        #
        # return newblist

    def play_all_strategies(self, Nrounds):
        print('---------------------------------- ')
        if Nrounds == 1:
            print('a single CHSH game')
        else:
            print(Nrounds, 'rounds of CHSH games, must win all of them')


        bestsofar = 0

        AliceStrategy = [0] * (Nrounds * pow(2, Nrounds))
        # AliceStrategy = [a, b, c,    d, e, f,    g, h, i, ...]
        # is made of blocks of length N (her composite answers to questions)
        # means answer abc to questions 000, cde to questions 001, fgh to questions 010, etc.
        # the b-th bit of the answer to question set Q is AliceStrategy[Q*Nrounds + b]
        for AliceStrategyCounter in range(0, pow(2, len(AliceStrategy))):
            AliceStrategy = self.binary_add1(AliceStrategy,(AliceStrategyCounter+1)%(self.n_questions*self.n_games))
            # AliceStrategy = self.binary_add1(AliceStrategy, (AliceStrategyCounter))

            BobStrategy = [0] * (Nrounds * pow(2, Nrounds))
            # same encoding for Bob's strategy [ans to 000, ans to 001, ans to 010, ...]
            for BobStrategyCounter in range(0, pow(2, len(AliceStrategy))):
                BobStrategy = self.binary_add1(BobStrategy,(BobStrategyCounter+1)%(self.n_questions*self.n_games))
                # BobStrategy = self.binary_add1(BobStrategy, (BobStrategyCounter))

                winning_probability = self.evaluate_parallelCHSH(AliceStrategy, BobStrategy, Nrounds, 0)  # the last 0 is no printout
                # print('tested strategies (',AliceStrategyCounter,BobStrategyCounter,'). win:',winning_probability)
                # print('..........................................')
                if winning_probability > bestsofar:
                    bestsofar = winning_probability
                    print('--------------------- best strategy candidate ------------')
                    print(bestsofar)
                    print(AliceStrategy, 'Alice strategy #', AliceStrategyCounter)
                    print(BobStrategy, 'Bob strategy #', BobStrategyCounter)
                    print('----------------------------------------------------------')

        print('maximum win probability: ', bestsofar)


if __name__ == '__main__':
    # go over all strategies
    # Alice's strategy is: for each set of her questions, give an answer
    # there are N questions for her, so she needs how to answwr 2^N different questions
    # each answer is an N bit string again, which means she has
    # (2^N)^(2^N) choices in her strategy!!!
    # TOO many strategies to try!
    # for N=1, she needs to decide on a 0 answer and a 1 answer...
    #                    4 strategies = 2^2
    #                    her (0->0, 1->0) (0->0, 1->1) (0->1, 1->0) (0->1, 1->1)
    # for N=2, she needs to decide on a 00, 01, 10, and 11 answer of length 2
    #                    4^4 = 256 strategies
    # for N=3, she needs to decide on answers to 000, 001, 010, 011, ..., 111
    #                   the answers have length 3 (8 possible answers in each)
    #                    8^8 = 2^24 = 16777216 strategies

    # the best 1-round strategy gives 0.75 = 3/4
    # see it here:
    import time

    start = time.time()
    # env = Environment(n_questions=2, n_games=1)
    # env.evaluate_parallelCHSH([1, 1],[1, 1],1,1)

    env = Environment(n_questions=2, n_games=2)
    env.play_all_strategies(2)

    print(time.time() - start)
    # the best 2-round strategy gives 0.625 = 10/16
    # see it here:
    env = Environment(n_questions=2, n_games=3)
    # env.evaluate_parallelCHSH([1, 0, 0, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 1, 1, 0],2,1)


    env.play_all_strategies(3)