# Bachelor-Thesis

Webpage:
https://janpastorek.github.io/Bachelor-Thesis/

Few words towards this project:

The first goal was to build a scalable non local game environment with databse.

The second goal was to build a reinforcement learning agent that will be able to search the non local game, learn the paths to bigger winning probability (CHSH value).

NonLocalGame provides API for doing operations on versions of nonlocalenvironments.

There are so far five versions of non local environments.

CHSHPrototype - is just a prototype that uses fixed gates(path) to illustrate how CHSH game (basic nonlocal game) works.

NlgDiscreteStatesActions - is fully functioning nonlocal game discrete environment optimized for discrete states and actions to be used on this environment.

NlgDiscreteTensorflow - is right now not functioning and will probably be deleted.

NlgDiscreteSortActions - always sorts actions used on the environment.

NlgGeneticOptimalization - is fully functioning nonlocal game environment where Genetic algorithm optimizes input rotation gates to learn the best possible way to maximize win_accuracy and therefore CHSH value. (could be set also for complex gates)

NlgContinuousOptimalization - environment uses NlgGeneticOptimalization to optimize gates that the agent chose each step.

I built two Reinforcement learning agents to search these environments:

Basic REGRESSION agent
DQN agent

On top of those I built Genetic algorithm that is able to optimalize agents hyperparameters (and also choose the best reward function)
