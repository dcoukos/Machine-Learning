'''
    This file contains all the models developed.

'''
from surprise import SVD
from surprise import accuracy


def svd(trainset, testset):
    # TODO  write code to optimize parameters

    '''code for checking if things are already saved
    from github

        modelname = 'svd'
        # Check if predictions already exist
        if is_already_predicted(modelname):
            return

    '''
    print('Running SVD model.')
    algo = SVD()
    algo.fit(trainset)
    tr_predictions = algo.test(trainset.build_testset())
    print('RMSE on training set: ', accuracy.rmse(tr_predictions, verbose=False))

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    print('RMSE on test set: ', rmse)
    return predictions
