import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import SVD
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from data_processing import (open_file, parse_as_dataset, parse_as_trainset,
                             write_predictions)

# TODO: Experiment with how to increase the accuracy of low count hits.
# TODO: Extract predictions from SVD algorithm


def rmse(pred_, act_):
    '''Global element-wise rmse calculation.'''
    prediction = pred_[act_.nonzero()].flatten()
    actual = act_[act_.nonzero()].flatten()
    return np.sqrt(mean_squared_error(prediction, actual))


# -------------------------      MAIN       -----------------------------------

# For reproducibility
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

data = open_file('data/data_train.csv')
ratings = parse_as_trainset(data)



algo = SVD()
# TODO: how to get predictions from CV?
predictions = cross_validate(algo, ratings, measures=['RMSE'], cv=5,
                             verbose=True)

# TODO: create test set.
# total_predictions = pd.DataFrame()
# TODO: implement crossvalidation.


kf = KFold(5)
total_predictions = pd.DataFrame()
for trainset, testset in kf.split(ratings):
    algo.fit(trainset)
    predictions = algo.test(testset)

    accuracy.rmse(predictions, verbose=True)
    predictions = pd.DataFrame(predictions)
    if total_predictions.empty is True:
        total_predictions = predictions
    else:
        total_predictions = pd.concat([total_predictions, predictions], axis=0)
