'''
    This file contains all the models developed.

'''
import numpy as np
from surprise import SVD
from surprise import accuracy
from helpers import (already_predicted, dump_predictions, dump_algo,
                     better_rmse, load_predictions)


def svd(trainset, testset, fullset, force_run=False):
    # TODO:  write code to optimize parameters for svd algo.
    # TODO: figure out how to minimize contamination!!
    '''code for checking if things are already saved
    from github

        modelname = 'svd'
        # Check if predictions already exist
        if is_already_predicted(modelname):
            return

    '''
    if not already_predicted('SVD') or force_run:


        # ------------ TESTING ---------------
        from data_processing import (open_file, parse_as_dataset, write_predictions)
        from models import svd
        from helpers import set_random
        set_random()
        data = open_file('data/data_train.csv')
        ratings = parse_as_dataset(data)
        ratings.split(n_folds=8)
        trainset, testset = next(ratings.folds())
        fullset = ratings
        #_______________ END __________________
        print('Running SVD model.')
        algo = SVD()
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

        if not already_predicted('SVD'):
            dump_predictions(predictions, 'SVD', rmse)
            dump_algo(algo, 'SVD', rmse)
        elif better_rmse('SVD', rmse):
            dump_predictions(predictions, 'SVD', rmse)
            dump_algo(algo, 'SVD', rmse)
        else:
            print('Predictions and algorithm not saved. Performance '
                  'inferior to existing model.')
    else:
        predictions = load_predictions('SVD')


    return predictions
