'''
    This file contains all the models developed.

'''
import numpy as np
from surprise import (SVD, KNNBasic, accuracy, BaselineOnly, SlopeOne, CoClustering)
from data_processing import open_file, parse_as_dataset


def svd(trainset, testset, fullset, labels,
        n_factors=5, n_epochs=40, lr_all=0.05, reg_all=0.02,
        train_as_test=False, testing=False):
    """Implements Single Value Decomposition .

    Parameters
    ----------
    n_factors : int
        Number of factors in matrix factorization.
    n_epochs : int
        Number of epochs through which to iterate.
    lr_all : float
        The learning rate for movies and usersdetermines how quickly the
        algorithm converges.
    reg_all : float
        Regularization term for movies and users.
    train_as_test: boolean
        A True value indicates to return predictions for the training set,
        which are used to determine coefficients for blending
    testing : boolean
        A True value indicates to return predictions for a test set, which are
        used to determine coefficient for blending

    Returns
    -------
    predictions, rmse: surprise.dataset, numpy.float64
        Set of rating predictions, and the root mean squared error of the model
        compared against the test set.
    """

    print('Running SVD model.')
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all,
               reg_all=reg_all)
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    tr_rmse = accuracy.rmse(tr_predictions, verbose=False)
    print('RMSE on training set: ', tr_rmse)
    if train_as_test:
        return tr_predictions, tr_rmse
    # This is the test evaluation.
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    if testing:
        return predictions, rmse
    print('RMSE on test set: ', rmse)
    print('Generating predictions')
    algo.fit(fullset)
    predictions = algo.test(labels.build_full_trainset().build_testset())
    rmse = np.around(rmse, 5)

    return predictions, rmse


def user_knn(trainset, testset, fullset, labels,
             min_support=10, range=40, train_as_test=False, testing=False):
    """Implements a k nearest neighbors algorithm from surprise,
    calculates similarities between users, using Pearon's coefficient.

    Parameters
    ----------
    min_support : int
        The minimum number of valid neighbors required in order to approximate
        neighborhood.
    range : int
        Maximum number of neighbors in a neighborhood.
    train_as_test: boolean
        A True value indicates to return predictions for the training set,
        which are used to determine coefficients for blending
    testing : boolean
        A True value indicates to return predictions for a test set, which are
        used to determine coefficient for blending

    Returns
    -------
    predictions, rmse: surprise.dataset, numpy.float64
        Set of rating predictions, and the root mean squared error of the model
        compared against the test set.
    """
    print('Running user-KNN model.')
    model_parameters = {
        'name': 'pearson',
        'user_based': False,
        'min_support': min_support
    }
    algo = KNNBasic(k=range, sim_options=model_parameters)
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    tr_rmse = accuracy.rmse(tr_predictions, verbose=False)
    print('RMSE on training set: ', tr_rmse)
    if train_as_test:
        return tr_predictions, tr_rmse
    # This is the test evaluation.
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    if testing:
        return predictions, labels
    print('RMSE on test set: ', rmse)
    print('Generating predictions')
    algo.fit(fullset)
    predictions = algo.test(labels.build_full_trainset().build_testset())
    rmse = np.around(rmse, 5)

    return predictions, rmse


def movie_knn(trainset, testset, fullset, labels,
              min_support=10, range=40, train_as_test=False, testing=False):
    """Implements a k nearest neighbors algorithm from surprise,
    calculates similarities between items, using Pearon's coefficient.

    Parameters
    ----------
    min_support : int
        The minimum number of valid neighbors required in order to approximate
        neighborhood.
    range : int
        Maximum number of neighbors in a neighborhood.
    train_as_test: boolean
        A True value indicates to return predictions for the training set,
        which are used to determine coefficients for blending
    testing : boolean
        A True value indicates to return predictions for a test set, which are
        used to determine coefficient for blending

    Returns
    -------
    predictions, rmse: surprise.dataset, numpy.float64
        Set of rating predictions, and the root mean squared error of the model
        compared against the test set.
    """
    print('Running movie-KNN model.')
    model_parameters = {
      'name': 'pearson',
      'user_based': False,
      'min_support': min_support
    }
    algo = KNNBasic(k=range, sim_options=model_parameters)
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    tr_rmse = accuracy.rmse(tr_predictions, verbose=False)
    print('RMSE on training set: ', tr_rmse)
    if train_as_test:
        return tr_predictions, tr_rmse
    # This is the test evaluation.
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    if testing:
        return predictions, rmse
    print('RMSE on test set: ', rmse)
    print('Generating predictions')
    algo.fit(fullset)
    predictions = algo.test(labels.build_full_trainset().build_testset())
    rmse = np.around(rmse, 5)
    return predictions, rmse


def baseline(trainset, testset, fullset, labels,
             train_as_test=False, testing=False):
    """Implements the baseline model from the surprise package. Only the global
    mean, a user's average deviation from the global mean, and an item's
    average deviation from the global mean are considered.

    Parameters
    ----------
    train_as_test: boolean
        A True value indicates to return predictions for the training set,
        which are used to determine coefficients for blending
    testing : boolean
        A True value indicates to return predictions for a test set, which are
        used to determine coefficient for blending

    Returns
    -------
    predictions, rmse: surprise.dataset, numpy.float64
        Set of rating predictions, and the root mean squared error of the model
        compared against the test set.
    """

    print('Running baseline model.')
    algo = BaselineOnly()
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    tr_rmse = accuracy.rmse(tr_predictions, verbose=False)
    print('RMSE on training set: ', tr_rmse)
    if train_as_test:
        return tr_predictions, tr_rmse
    # This is the test evaluation.
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    if testing:
        return predictions, rmse
    print('RMSE on test set: ', rmse)
    print('Generating predictions')
    algo.fit(fullset)
    predictions = algo.test(labels.build_full_trainset().build_testset())
    rmse = np.around(rmse, 5)
    return predictions, rmse


def slope_one(trainset, testset, fullset, labels,
              train_as_test=False, testing=False):
    """Implements the slope one algorithm from the surprise package. This is a
    fast, simple algorithm based of single parameter regressions of the form
    f(x) = x + b.

    Parameters
    ----------
    train_as_test: boolean
        A True value indicates to return predictions for the training set,
        which are used to determine coefficients for blending
    testing : boolean
        A True value indicates to return predictions for a test set, which are
        used to determine coefficient for blending

    Returns
    -------
    predictions, rmse: surprise.dataset, numpy.float64
        Set of rating predictions, and the root mean squared error of the model
        compared against the test set.
    """

    print('Running Slope-One model.')
    algo = SlopeOne()
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    tr_rmse = accuracy.rmse(tr_predictions, verbose=False)
    print('RMSE on training set: ', tr_rmse)
    if train_as_test:
        return tr_predictions, tr_rmse
    # This is the test evaluation.
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('RMSE on test set: ', rmse)
    if testing:
        return predictions, rmse
    print('Generating predictions')
    algo.fit(fullset)
    predictions = algo.test(labels.build_full_trainset().build_testset())
    rmse = np.around(rmse, 5)
    return predictions, rmse


def co_clustering(trainset, testset, fullset, labels,
                  n_clstr_usr=3, n_clstr_mv=3, train_as_test=False,
                  testing=False):
    """Implements the co-clustering algorithm from the surprise package. Users
    and movies are divided into clusters, from which predictions are made
    using the global average, the average of an item-user's co-cluster, the
    average of an item's cluster, and the average of the user's cluster.
    movie's

    Parameters
    ----------
    n_clstr_usr : int
        Number of clusters into which to divide the users.
    n_clstr_mv : int
        Number of clusters into which to divide the items.
    train_as_test: boolean
        A True value indicates to return predictions for the training set,
        which are used to determine coefficients for blending
    testing : boolean
        A True value indicates to return predictions for a test set, which are
        used to determine coefficient for blending

    Returns
    -------
    predictions, rmse: surprise.dataset, numpy.float64
        Set of rating predictions, and the root mean squared error of the model
        compared against the test set.

    """
    print('Running Co-Clustering model.')
    algo = CoClustering(n_cltr_u=n_clstr_usr, n_cltr_i=n_clstr_mv)
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    tr_rmse = accuracy.rmse(tr_predictions, verbose=False)
    print('RMSE on training set: ', tr_rmse)
    if train_as_test:
        return tr_predictions, tr_rmse
    # This is the test evaluation.
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('RMSE on test set: ', rmse)
    if testing:
        return predictions, rmse
    print('Generating predictions')
    algo.fit(fullset)
    predictions = algo.test(labels.build_full_trainset().build_testset())
    rmse = np.around(rmse, 5)
    return predictions, rmse
