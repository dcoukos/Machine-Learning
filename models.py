'''
    This file contains all the models developed.

'''
import numpy as np
from surprise import (SVD, KNNBasic, accuracy, BaselineOnly, SlopeOne, CoClustering)
from data_processing import open_file, parse_as_dataset
from surprise.model_selection import train_test_split


def svd(n_factors=5, n_epochs=40, lr_all=0.05, reg_all=0.02, testing=False):
    '''Implements the single vale decomposition from surprise.
    '''
    data = open_file('data/data_train.csv')
    labels = open_file('data/data_test.csv')
    ratings = parse_as_dataset(data)
    labels = parse_as_dataset(labels)
    data = parse_as_dataset(data)
    trainset, testset = train_test_split(ratings, test_size=0.2)
    fullset = data.build_full_trainset()

    print('Running SVD model.')
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all,
               reg_all=reg_all)
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    tr_rmse = accuracy.rmse(tr_predictions, verbose=False)
    print('RMSE on training set: ', tr_rmse)
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
    if not testing:
        if not already_predicted('SVD'):
            dump_predictions(predictions, 'SVD', rmse)
            dump_algo(algo, 'SVD', rmse)
        elif better_rmse('SVD', rmse):
            dump_predictions(predictions, 'SVD', rmse)
            dump_algo(algo, 'SVD', rmse)
        else:
            print('Predictions and algorithm not saved. Performance '
                  'inferior to existing model.')
    if testing:
        return algo, predictions, rmse, tr_rmse
    return predictions, rmse


def user_knn(min_support=10, algorithm='basic', testing=False, range=40, testing=False):
    data = open_file('data/data_train.csv')
    labels = open_file('data/data_test.csv')
    ratings = parse_as_dataset(data)
    labels = parse_as_dataset(labels)
    data = parse_as_dataset(data)
    trainset, testset = train_test_split(ratings, test_size=0.2)
    fullset = data.build_full_trainset()
    print('Running user-KNN model.')
    model_parameters = {
        'name': 'pearson',
        'user_based': False,
        'min_support': min_support
    }
    algo = KNNBasic(k=range, sim_options=model_parameters)
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    print('RMSE on training set: ', accuracy.rmse(tr_predictions,
          verbose=False))
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


def movie_knn(min_support=10, algorithm='basic', testing=False, range=40, testing=False):
    data = open_file('data/data_train.csv')
    labels = open_file('data/data_test.csv')
    ratings = parse_as_dataset(data)
    labels = parse_as_dataset(labels)
    data = parse_as_dataset(data)
    trainset, testset = train_test_split(ratings, test_size=0.2)
    fullset = data.build_full_trainset()
    print('Running movie-KNN model.')
    model_parameters = {
      'name': 'pearson',
      'user_based': False,
      'min_support': min_support
    }
    if algorithm is 'basic':
        algo = KNNBasic(k=range, sim_options=model_parameters)
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    print('RMSE on training set: ', accuracy.rmse(tr_predictions,
          verbose=False))
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


def baseline(testing=False):
    data = open_file('data/data_train.csv')
    labels = open_file('data/data_test.csv')
    ratings = parse_as_dataset(data)
    labels = parse_as_dataset(labels)
    data = parse_as_dataset(data)
    trainset, testset = train_test_split(ratings, test_size=0.2)
    fullset = data.build_full_trainset()

    print('Running baseline model.')
    algo = BaselineOnly()
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    print('RMSE on training set: ', accuracy.rmse(tr_predictions,
          verbose=False))
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


def slope_one(testing=False):
    data = open_file('data/data_train.csv')
    labels = open_file('data/data_test.csv')
    ratings = parse_as_dataset(data)
    labels = parse_as_dataset(labels)
    data = parse_as_dataset(data)
    trainset, testset = train_test_split(ratings, test_size=0.2)
    fullset = data.build_full_trainset()

    print('Running Slope-One model.')
    algo = SlopeOne()
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    print('RMSE on training set: ', accuracy.rmse(tr_predictions,
          verbose=False))
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


def co_clustering(n_clstr_usr=3, n_clstr_mv=3, testing=False):
    data = open_file('data/data_train.csv')
    labels = open_file('data/data_test.csv')
    ratings = parse_as_dataset(data)
    labels = parse_as_dataset(labels)
    data = parse_as_dataset(data)
    trainset, testset = train_test_split(ratings, test_size=0.2)
    fullset = data.build_full_trainset()

    print('Running Co-Clustering model.')
    algo = CoClustering(n_cltr_u=n_clstr_usr, n_cltr_i=n_clstr_mv)
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    print('RMSE on training set: ', accuracy.rmse(tr_predictions,
          verbose=False))
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
