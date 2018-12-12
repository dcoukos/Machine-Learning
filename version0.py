import csv
import os
import pandas as pd
import numpy as np
import numpy.linalg as linalg
from sklearn.metrics import mean_squared_error
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import SVD
from surprise.model_selection import cross_validate
from collections import Counter

# TODO: Experiment with how to increase the accuracy of low count hits.
# TODO: Extract predictions from SVD algorithm


def open_file(path):
    '''Open csv, and return list of lines.'''
    with open(path, 'r') as f:
        return f.read().splitlines()[1:]


def parse_as_dataframe(data_list):
    '''Returns DataFrame containing items, users, and associated ratings'''
    ratings_dict = {'item': [], 'user': [], 'rating': []}

    for line in data_list:
        row, col, rating = parse_review_line(line)
        ratings_dict['item'].append(row)
        ratings_dict['user'].append(col)
        ratings_dict['rating'].append(rating)
    return pd.DataFrame(ratings_dict, columns=['item', 'user', 'rating'])


def parse_as_trainset(data_list):
    data = parse_as_dataframe(data_list)
    dataset = Dataset.load_from_df(data, Reader())
    return dataset.build_full_trainset()


def parse_review_line(line):
    '''Extracts item tag, user, and rating from string'''
    pos, rating = line.split(',')
    row, col = pos.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return row, col, rating


def remove_tail(ratings, min_count=10):
    '''Removes data for items, users which are present less than the min_count
    '''
    counts_item = Counter(ratings['item'])
    counts_item = dict(counts_item)
    valid_items = {key: value for (key, value) in counts_item.items()
                   if (value > min_count-1)}
    valid_ratings = pd.DataFrame()
    valid_ratings = ratings.loc[ratings['item'].isin(valid_items.keys())]
    counts_user = Counter(valid_ratings['user'])
    counts_user = dict(counts_user)
    valid_users = {key: value for (key, value) in counts_user.items()
                   if (value > min_count-1)}
    valid_ratings = valid_ratings.loc[ratings['user'].isin(valid_users.keys())]
    return valid_ratings


def rmse(pred_, act_):
    '''Global element-wise rmse calculation.'''
    prediction = pred_[act_.nonzero()].flatten()
    actual = act_[act_.nonzero()].flatten()
    return np.sqrt(mean_squared_error(prediction, actual))


# -------------------------      MAIN       -----------------------------------

data = open_file('data_train.csv')
ratings = parse_as_trainset(data)


algo = SVD()
algo.fit(ratings)  # To Know: fit before CV?
predictions = cross_validate(algo, ratings, measures=['RMSE'], cv=5,
                             verbose=True)


# TODO: create test set.
total_predictions = pd.DataFrame()
# TODO: implement crossvalidation.
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)

    accuracy.rmse(predictions, verbose=True)
    predictions = pd.DataFrame(predictions)
    if total_predictions.empty is True:
        total_predictions = predictions
    else:
        total_predictions = pd.concat([total_predictions, predictions], axis = 0)


predictions = [(int(u), int(i), rat) for (u, i, _, rat, _)
               in total_predictions.values]
header = 'Id,Prediction\n'
data = [header]
for pred in predictions:
    data.append('r{u}_c{i},{r}\n'.format(u=pred[0], i=pred[1], r=pred[2]))

path = os.path.join('predictions.csv')
fp = open(path, 'w')
fp.writelines(data)
fp.close()



total_predictions

total_predictions.values
