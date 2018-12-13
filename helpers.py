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
    n_movies, n_users = trainset.get_shape()

    user_features = np.random.rand(n_features, n_users)
    item_features = np.random.rand(n_features, n_movies)

    item_nnz = trainset.getnnz(axis=1)
    item_sum = trainset.sum(axis=1)

    for ind in range(n_movies):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features


def dump_predictions(data, model, rmse):
    '''Pickles predictions'''
    filepath = os.path.join('pickles',)
    pickle.dump(data, )


def dump_algo(name, rmse):
    '''Pickles characteristic parameters of algorithm'''



def already_predicted(modelname):
    basepath = os.path.join('pickles')
    if glob(os.path.join(basepath, modelname + '_*')):
        return True
    else:
        return False
