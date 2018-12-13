'''
    This file contains functions for preprocessing data, cleaning the data,
    and parsing it to different typesself.

    (open_file, parse_as_dataframe, parse_as_dataset,
    parse_as_trainset, parse_review_line, remove_tail)
'''
import os
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from collections import Counter


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


def parse_as_dataset(data_list):
    data = parse_as_dataframe(data_list)
    return Dataset.load_from_df(data, Reader())


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


def write_predictions(filename, df_predictions):
    '''Saves predictions in dataframe to file defined by filepath.'''
    predictions = [(int(i), int(u), int(np.round(rat))) for (i, u, _, rat, _)
                   in df_predictions.values]
    header = 'Id,Prediction\n'
    data = [header]
    for pred in predictions:
        data.append('r{0}_c{1},{2}\n'.format(*pred))

    print(check_labels(data))

    path = os.path.join('predictions/', filename)
    fp = open(path, 'w')
    fp.writelines(data)
    fp.close()


def check_labels(data):
    print('Checking labels')
    original = open_file('data/data_train.csv')
    ori_labels = []
    for line in original:
        label, _ = line.split(',')
        ori_labels.append(label)

    pred_labels = []
    for line in data[1:]:
        label, _ = line.split(',')
        pred_labels.append(label)

    differences = set(ori_labels) - set(pred_labels)
    intersection = set(ori_labels) & set(pred_labels)
    if len(intersection) == 1176952:
        return True
    else:
        return differences
