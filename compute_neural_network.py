import argparse
import matplotlib
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
from mlrose_hiive.algorithms.decay import GeomDecay, ExpDecay, ArithDecay
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, train_test_split, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

RANDOM_STATE = 3

def load_pima():
    df = pd.read_csv('pima/pima-indians-diabetes.csv')
    df.columns = [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome' ]
    data = df[ [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ] ]
    target = df[ [ 'Outcome' ] ]
    return data, target

# from sklearn.neural_network import MLPClassifier
# this function was taken and modified from the sklearn documentation! 
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.suptitle( title )

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")

    print('done plotting convergence')

    plt.show()

    # Plot fit_time vs score
    plt.grid()
    plt.plot(fit_times_mean, test_scores_mean, 'o-')
    plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    plt.xlabel("fit_times")
    plt.ylabel("Score")
    plt.title(title)

    plt.show()
    return plt

def make_plot( fig, axes, clf, title, X, y ):
    cv = ShuffleSplit( n_splits = 100, test_size = 0.2, random_state = 0 )
    plot_learning_curve( clf, title, X, y, axes = axes, ylim=( 0.2, 1.01 ), cv = cv, n_jobs = 4 )

class NNTrainer:
    def __init__(self):
        # data = load_digits()
        X, y = load_pima()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = RANDOM_STATE)
        # X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2, random_state = RANDOM_STATE)
        
        # Normalize feature data
        scaler = MinMaxScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        # One hot encode target values
        one_hot = OneHotEncoder()
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        self.y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
        self.y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
    
    def get_score(self, network):
        network.fit(self.X_train_scaled, self.y_train_hot)
        # Predict labels for train set and assess accuracy
        y_train_pred = network.predict(self.X_train_scaled)
        y_train_accuracy = accuracy_score(self.y_train_hot, y_train_pred)
        print(y_train_accuracy)
        # Predict labels for test set and assess accuracy
        y_test_pred = network.predict(self.X_test_scaled)
        y_test_accuracy = accuracy_score(self.y_test_hot, y_test_pred)
        print(y_test_accuracy)
        return y_test_accuracy

    def run(self, algorithm, plot = False):
        print( 'Optimizing weights with %s' % algorithm )
        assert(algorithm in ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'])
        # Initialize neural network object and fit object - attempt 1
        best_score = 0
        best_val = None
        for val in [10 * i for i in range(1, 5)]:
            print( 'Trying with %s' % val )
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                            algorithm = algorithm, 
                                            pop_size = 200,
                                            mutation_prob = 0.1,
                                            max_iters = 100, bias = True, is_classifier = True, 
                                            learning_rate = 0.001, early_stopping = True, 
                                            clip_max = 5, max_attempts = 100, random_state = RANDOM_STATE)
            score = self.get_score( nn_model1 )
            if score > best_score:
                best_score = score
                best_val = val
        print( 'Best score: %s, with %s' % (best_score, best_val) )
        if plot:
            X, y = self.X_train_scaled, self.y_train_hot
            fig, axes = plt.subplots( 1, 1, figsize = ( 10, 15 ) )
            make_plot(fig, axes, nn_model1, 'Genetic Algorithm', X, y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--learner', action = 'store', dest = 'learner', required = True )
    parser.add_argument( '--plot', action = 'store_true', dest = 'plot', required = False)
    args = parser.parse_args()
    
    nn = NNTrainer()
    nn.run(args.learner, plot = args.plot)