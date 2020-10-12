import argparse
import matplotlib
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
from mlrose_hiive.algorithms.decay import GeomDecay, ArithDecay, ExpDecay
import numpy as np

COMPLEXITY = 40
MAX_ATTEMPTS = 100

fitness_evals = []
def get_custom_fitness(fitness):
    def counting_fitness(state):
        global fitness_evals
        score = fitness.evaluate(state)
        fitness_evals.append( score )
        return score
    return counting_fitness

def get_fitness_eval_curve():
    fitness_eval_x = []
    fitness_eval_y = []
    maximum = 0
    for i in range(len(fitness_evals)):
        if fitness_evals[i] > maximum:
            maximum = fitness_evals[i]
            fitness_eval_x.append(i)
            fitness_eval_y.append(fitness_evals[i])
    return fitness_eval_x, fitness_eval_y

class OptimizationProblem:
    def __init__(self, complexity, fitness_fn):
        self.problem = mlrose.DiscreteOpt(length = complexity, fitness_fn = fitness_fn)
        self.max_attempts = MAX_ATTEMPTS
    
    def solveAndAverage(self, alg, **kwargs):
        scores = []
        curves = []
        max_score = 0
        best_state = None
        # THIS MAY TAKE A WHILE, CHANGE THIS RANGE TO range(1,5) FOR A FASTER BUT ROUGHER RUN
        for random_seed in range(1, 21):
            state, score, curve = alg(**kwargs, random_state = random_seed)
            scores.append(score)
            curves.append(curve)
            if score > max_score:
                max_score = score
                best_state = state
        average_score = np.mean(scores)
        average_curve = [np.mean([c[i] for c in curves]) for i in range(min(len(c) for c in curves))]

        return best_state, average_score, average_curve
    
    def solveWithHillClimbing(self):
        return self.solveAndAverage(alg = mlrose.random_hill_climb,
                                    problem = self.problem,
                                    max_attempts = self.max_attempts,
                                    restarts = 30,
                                    curve = True)
    
    def solveWithSimulatedAnnealing(self):
        return self.solveAndAverage(alg = mlrose.simulated_annealing,
                                    problem = self.problem,
                                    schedule = ExpDecay(),
                                    max_attempts = self.max_attempts,
                                    curve = True)
    
    def solveWithGeneticAlg(self):
        return self.solveAndAverage(alg = mlrose.genetic_alg,
                                    pop_size = 160,
                                    mutation_prob = 0.4,
                                    problem = self.problem,
                                    max_attempts = 10,
                                    curve = True)
    
    def solveWithMimic(self):
        return self.solveAndAverage(alg = mlrose.mimic,
                                    pop_size = 160,
                                    keep_pct = 0.3,
                                    problem = self.problem,
                                    max_attempts = self.max_attempts,
                                    curve = True)

def fitnessVsIterations(fitness, problem_name):
    OP = OptimizationProblem(complexity = COMPLEXITY, fitness_fn = fitness)
    global fitness_evals
    fitness_evals = []
    best_state, best_fitness, rhc_curve = OP.solveWithHillClimbing()
    print( 'Best state found by RHC: %s' % best_state )
    print( 'Best fitness found by RHC: %s' % best_fitness )
    x, y = get_fitness_eval_curve()
    plt.plot(x, y)
    fitness_evals = []
    best_state, best_fitness, annealing_curve = OP.solveWithSimulatedAnnealing()
    print( 'Best state found by Simulated Annealing: %s' % best_state )
    print( 'Best fitness found by Simulated Annealing: %s' % best_fitness )
    x, y = get_fitness_eval_curve()
    plt.plot(x, y)
    fitness_evals = []
    best_state, best_fitness, genetic_curve = OP.solveWithGeneticAlg()
    print( 'Best state found by Genetic Alg: %s' % best_state )
    print( 'Best fitness found by Genetic Alg: %s' % best_fitness )
    x, y = get_fitness_eval_curve()
    plt.plot(x, y)
    fitness_evals = []
    best_state, best_fitness, mimic_curve = OP.solveWithMimic()
    print( 'Best state found by MIMIC: %s' % best_state )
    print( 'Best fitness found by MIMIC: %s' % best_fitness )
    x, y = get_fitness_eval_curve()
    plt.plot(x, y)

    plt.legend(['Random Hill Climbing', 'Simulated Annealing', 'Genetic Alg', 'MIMIC'])
    plt.suptitle(f'{problem_name}: Fitness Vs Fitness Function Evaluations')
    plt.ylabel('Fitness Score')
    plt.xlabel('Evaluations')
    plt.show()

    plt.plot(rhc_curve)
    plt.plot(annealing_curve)
    plt.plot(genetic_curve)
    plt.plot(mimic_curve)

    plt.legend(['Random Hill Climbing', 'Simulated Annealing', 'Genetic Alg', 'MIMIC'])
    plt.suptitle(f'{problem_name}: Fitness Vs Iterations')
    plt.ylabel('Fitness Score')
    plt.xlabel('Iterations')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--problem', action = 'store', dest = 'problem', required = True )
    args = parser.parse_args()

    if args.problem == 'onemax':
        fitness = mlrose.OneMax()
        problem_name = 'Onemax'
    elif args.problem == 'four_peaks':
        fitness = mlrose.FourPeaks(t_pct = 0.15)
        problem_name = 'Four Peaks'
    elif args.problem == 'kcolor':
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        fitness = mlrose.MaxKColor(edges)
        # problem = mlrose.DiscreteOpt(length = 5, fitness_fn = fitness)
    elif args.problem == 'flipflop':
        fitness = mlrose.FlipFlop()
        problem_name = 'Flip Flop'
    else:
        raise RuntimeError("Invalid problem argument")

    custom_fitness_function = get_custom_fitness(fitness)
    custom_fitness = mlrose.CustomFitness(custom_fitness_function)

    fitnessVsIterations(custom_fitness, problem_name)