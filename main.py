import os
from pandas import DataFrame
from data_processing import (open_file, parse_as_dataset, write_predictions)
from models import svd
from helpers import set_random

# TODO: Experiment with how to increase the accuracy of low count hits.
# TODO: Extract predictions from SVD algorithm



# -------------------------      MAIN       ----------------------------------

# For reproducibility
set_random()

data = open_file('data/data_train.csv')
ratings = parse_as_dataset(data)

# TODO: separate input data into training and test data.
ratings.split(n_folds=8)
trainset, testset = next(ratings.folds())

predictions = svd(trainset, testset, ratings)

write_predictions('svd.csv', DataFrame(predictions))
