'''
    This file contains all the models developed.

'''
from surprise import SVD
from surprise import accuracy
from helpers import already_predicted


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
    if not already_predicted('svd') or force_run:
        print('Running SVD model.')
        algo = SVD()
        algo.fit(trainset)
        tr_predictions = algo.test(trainset.build_testset())
        print('RMSE on training set: ', accuracy.rmse(tr_predictions, verbose=False))

        # This is the test evaluation.
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        print('RMSE on test set: ', rmse)
        print('Generating predictions')
        dump_algo(algo)
        predictions = algo.test(fullset.build_full_trainset().build_testset())
    else:


    return predictions
