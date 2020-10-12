import argparse
import matplotlib
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
from mlrose_hiive.algorithms.decay import GeomDecay, ArithDecay, ExpDecay
import numpy as np
import time

SIZE = 40

algs = {
    'hill_climbing': mlrose.random_hill_climb,
    'simulated_annealing': mlrose.simulated_annealing,
    'genetic_alg': mlrose.genetic_alg,
    'mimic': mlrose.mimic
}

fitness_evals = []

def get_score(alg, problem, params, num_seeds = 30):
    fitnesses = []
    fitness_curves = []
    best_fitness = 0
    best_state = None
    for random_seed in range(1,num_seeds+1):
        params.update({'random_state': random_seed, 'curve': True})

        global fitness_evals
        fitness_evals = []

        state, fitness, curve = alg(**params)
        if fitness > best_fitness:
            best_fitness = fitness
            best_state = state
        
        fitnesses.append(fitness)
        fitness_curves.append(curve)
    
    print(best_state)
    
    avg_fitness_curve = [ np.mean( [ c[i] for c in fitness_curves ] ) for i in range(min([len(c) for c in fitness_curves])) ]
    return np.mean(fitnesses), avg_fitness_curve

def get_hyperparam_score(alg, problem, params, num_seeds = 20):
    fitnesses = []
    for random_seed in range(1, num_seeds+1):
        params.update({'random_state': random_seed, 'curve': True})
        state, fitness, curve = alg(**params)
        fitnesses.append(fitness)

    return np.mean(fitnesses)

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

def get_final_score(learner, fitness, hyperparams):
    problem = mlrose.DiscreteOpt(length = SIZE, fitness_fn = fitness, maximize = True)
    params = hyperparams.copy()
    params.update({'problem': problem})

    start = time.process_time()
    score, curve = get_score(alg, problem, params)
    print( 'Time taken to learn:', time.process_time() - start )

    print( 'Average Best Score over 30 random seeds:', score )

    plt.plot( curve )
    plt.xlabel( 'Iterations' )
    plt.ylabel( 'Average Fitness Score' )
    plt.suptitle( learner )
    plt.show()

    x, y = get_fitness_eval_curve()
    plt.plot(x, y)
    plt.xlabel( 'Fitness Function Evaluations' )
    plt.ylabel( 'Fitness Score' )
    plt.suptitle( learner )
    plt.show()
    return score

def find_best_hyperparameter(alg, fitness, params, param_name, possible_values, show_graph = True):
    best_avg_score = 0
    best_val = None
    scores = []
    plt.xscale('log')
    names = []
    for v in possible_values:
        problem = mlrose.DiscreteOpt(length = SIZE, fitness_fn = fitness, maximize = True)
        problem.set_mimic_fast_mode( True )
        cur_params = params.copy()
        cur_params.update({param_name: v, 'problem': problem})
        score, curve = get_score(alg, problem, cur_params)
        plt.plot( curve )
        names.append(v)
    plt.xlabel( 'Iterations' )
    plt.ylabel( 'Average Fitness Score' )
    plt.legend( names )
    plt.suptitle( 'MIMIC Keep Pct: Four Peaks' )
    plt.show()

    # commented out is the mechanical way of finding hyperparams. I decided that taking the highest average score might not 
    # be taking everything into account (number of iterations taken to reach max, etc)
    # so I decided to choose by plotting the curves with each other (above)

    # score = get_hyperparam_score(alg, problem, cur_params)
    # scores.append(score)
    # if score > best_avg_score:
    #     best_avg_score = score
    #     best_val = v
    
    # if show_graph:
    #     plt.plot([type(v).__name__ for v in possible_values], scores)
    #     plt.xlabel( f'{param_name}' )
    #     plt.ylabel( 'Average Fitness Score' )
    #     plt.suptitle( f'{param_name}' )
    #     plt.show()

    return best_val        

def optimize_hyperparams(title, alg, fitness, params, hyperparams):
    best_hyperparams = {}
    for param_name, (possible_values, show_graph) in hyperparams.items():
        print( 'optimizing hyperparameter %s' % param_name )
        best_hyperparams[param_name] = find_best_hyperparameter(alg, fitness, params, param_name, possible_values, show_graph)
        print( 'optimal value found: %s' % best_hyperparams[param_name] )
    
    print('parameters optimized.')
    
    return best_hyperparams

MAX_ATTEMPTS = 100

def optimize_hill_climbing_hyperparams(fitness):
    params = { 'max_attempts': MAX_ATTEMPTS, 'restarts': 20 }
    return params

def optimize_simulated_annealing_hyperparams(fitness):
    params = { 'max_attempts': MAX_ATTEMPTS }
    hyperparams = { 'schedule': ([GeomDecay(), ArithDecay(), ExpDecay()], True) }
    return optimize_hyperparams( 'Simulated Annualing', mlrose.simulated_annealing, fitness, params, hyperparams )

def optimize_genetic_alg_hyperparams(fitness):
    params = { 'max_attempts': MAX_ATTEMPTS }
    hyperparams = { 'pop_size': ([i for i in [SIZE/2, SIZE, SIZE * 2, SIZE * 3, SIZE * 4]], True),
                    'pop_breed_percent': ([0.5 + 0.1 * i for i in range(5)], True),
                    'elite_dreg_ratio': ( [0.95 + 0.01 * i for i in range(5) ], True ),
                    'mutation_prob': ( [0.2 * i for i in range(1, 5)], True) }
    
    optimized = optimize_hyperparams( 'Genetic Alg', mlrose.genetic_alg, fitness, params, hyperparams )
    params.update(optimized)
    return params

def optimize_mimic_hyperparams(fitness):
    params = {}
    hyperparams = { 'pop_size': ( [ 100 + 20*i for i in range(11) ], True ),
                    'keep_pct': ( [ 0.1 + 0.02*i for i in range(11) ], True ) }
    
    optimized = optimize_hyperparams( 'MIMIC', mlrose.mimic, fitness, params, hyperparams )
    params.update(optimized)
    return params

def get_custom_fitness(fitness):
    def counting_fitness(state):
        global fitness_evals
        score = fitness.evaluate(state)
        fitness_evals.append( score )
        return score
    return counting_fitness

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--learner', action = 'store', dest = 'learner', required = True )
    parser.add_argument( '--problem', action = 'store', dest = 'problem', required = True )
    args = parser.parse_args()

    if args.problem == 'onemax':
        fitness = mlrose.OneMax()
    elif args.problem == 'four_peaks':
        fitness = mlrose.FourPeaks(t_pct = 0.15)
    elif args.problem == 'kcolor':
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        fitness = mlrose.MaxKColor(edges)
        problem = mlrose.DiscreteOpt(length = 5, fitness_fn = fitness)
        init_state = np.array([0, 1, 0, 1, 1])
    elif args.problem == 'flipflop':
        fitness = mlrose.FlipFlop()
    else:
        raise RuntimeError("Invalid problem argument")

    if args.learner == 'hill_climbing':
        hyperparams = optimize_hill_climbing_hyperparams(fitness)
    elif args.learner == 'simulated_annealing':
        hyperparams = optimize_simulated_annealing_hyperparams(fitness)
    elif args.learner == 'genetic_alg':
        hyperparams = optimize_genetic_alg_hyperparams(fitness)
    elif args.learner == 'mimic':
        hyperparams = optimize_mimic_hyperparams(fitness)
    else:
        raise RuntimeError("Invalid learner argument")

    print( 'found hyperparams:', hyperparams )

    custom_fitness_function = get_custom_fitness(fitness)
    custom_fitness = mlrose.CustomFitness(custom_fitness_function)

    alg = algs[args.learner]
    
    get_final_score(args.learner, custom_fitness, hyperparams)