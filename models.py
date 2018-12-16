'''
    This file contains all the models developed.

'''
import numpy as np
from surprise import (SVD, KNNBasic, accuracy)
from helpers import (already_predicted, dump_predictions, dump_algo,
                     better_rmse, load_predictions, load_algo, beep)


def svd(trainset, testset, fullset, n_factors=100, n_epochs=20, lr_all=0.005,
        reg_all=0.02, force=False, testing=False):
    # TODO:  write code to optimize parameters for svd algo.
    # TODO: figure out how to minimize contamination!!
    '''code for checking if things are already saved
    from github

        modelname = 'svd'
        # Check if predictions already exist
        if is_already_predicted(modelname):
            return

    '''
    if not already_predicted('SVD') or force or testing:
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
        print('RMSE on test set: ', rmse)
        print('Generating predictions')
        predictions = algo.test(fullset.build_full_trainset().build_testset())
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
            beep()
        if testing:
            return algo, predictions, rmse - tr_rmse
    else:
        print('Loading SVD model from file...')
        predictions, rmse = load_predictions('SVD')
    return predictions, rmse


def user_knn(trainset, testset, fullset, min_support=10, algorithm='basic',
             force=False):
    if not already_predicted('userKNN') or force:
        print('Running user-KNN model.')
        model_parameters = {
            'name': 'pearson',
            'user_based': True,
            'min_support': min_support
        }
        if algorithm is 'basic':
            algo = KNNBasic(sim_options=model_parameters)
        algo.fit(trainset)
        tr_predictions = algo.test(trainset.build_testset())
        print('RMSE on training set: ', accuracy.rmse(tr_predictions,
              verbose=False))
        # This is the test evaluation.
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        print('RMSE on test set: ', rmse)
        print('Generating predictions')
        predictions = algo.test(fullset.build_full_trainset().build_testset())
        rmse = np.around(rmse, 5)

        if not already_predicted('userKNN'):
            dump_predictions(predictions, 'userKNN', rmse)
            dump_algo(algo, 'userKNN', rmse)
        elif better_rmse('userKNN', rmse):
            dump_predictions(predictions, 'userKNN', rmse)
            dump_algo(algo, 'userKNN', rmse)
        else:
            print('Predictions and algorithm not saved. Performance '
                  'inferior to existing model.')
        beep()
    else:
        print('Loading user-KNN model from file...')
        predictions, rmse = load_predictions('userKNN')

    return predictions, rmse
