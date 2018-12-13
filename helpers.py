import os
import random
import pickle
import numpy as np
from glob import glob
from sklearn.metrics import mean_squared_error


def set_random(my_seed=0):
    '''Ensures model reproducibility.'''
    random.seed(my_seed)
    np.random.seed(my_seed)


def rmse(pred_, act_):
    '''Global element-wise rmse calculation.'''
    prediction = pred_[act_.nonzero()].flatten()
    actual = act_[act_.nonzero()].flatten()
    return np.sqrt(mean_squared_error(prediction, actual))


def initialize_MF(trainset, n_features):
    '''Initialize paramter matrices for matrix factorization'''
    n_movies, n_users = trainset.get_shape()

    user_features = np.random.rand(n_features, n_users)
    item_features = np.random.rand(n_features, n_movies)

    item_nnz = trainset.getnnz(axis=1)
    item_sum = trainset.sum(axis=1)

    for ind in range(n_movies):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features


def dump_predictions(predictions, modelname, rmse):
    """Pickles predictions

    Parameters
    ----------
    data :
        Predictions generated by model. May be dataframe, or data generated by
        surprise.
    modelname : string
    rmse : float
        rmse against test set.
    """
    rmse_ = rmse.astype('str').replace('.', '-')
    filepath = os.path.join('pickles', modelname + '_' + rmse_ + '.p')
    filehandle = open(filepath, 'wb')
    pickle.dump(predictions, filehandle)


def dump_algo(algo, modelname, rmse):
    '''Pickles characteristic parameters of algorithm

    Parameters
    ----------
    algo :
        Characteristic parameters representing the optimized model.
    modelname : string
    rmse : float
        rmse against test set.
    '''
    rmse_ = rmse.astype('str').replace('.', '-')
    filepath = os.path.join('pickles', 'algo_' + modelname + '_' + rmse_ +
                            '.p')
    filehandle = open(filepath, 'wb')
    pickle.dump(algo, filehandle)


def load_predictions(modelname):
    filepath = os.path.join('pickles', modelname + '_*' + '.p')
    return pickle.load(open(filepath, 'rb'))


def load_algo(name):
    filepath = os.path.join('pickles', 'algo_' + name + '_*' + '.p')
    return pickle.load(open(filepath, 'rb'))


def better_rmse(modelname, rmse):
    '''Checks if pickled predictions, or model have better rmse than current'''
    basepath = os.path.join('pickles', modelname + '_')
    filename = os.path.relpath(basepath)
    _, pickled_rmse = filename.split('_')
    pickled_rmse = pickled_rmse.replace('-', '.')
    if rmse > float(pickled_rmse):
        return True
    else:
        return False


def already_predicted(modelname):
    '''Checks if movie ratings have been predicted already using a model'''
    basepath = os.path.join('pickles')
    if glob(os.path.join(basepath, modelname + '_*')):
        return True
    else:
        return False
