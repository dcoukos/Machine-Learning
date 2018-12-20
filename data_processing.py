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
from surprise.model_selection import train_test_split
from surprise import BaselineOnly


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


def write_predictions(modelname, rmse, df_predictions):
    '''Saves predictions in dataframe to file defined by filepath.'''
    if already_written(modelname, rmse):
        return 'already_written'
    else:
        predictions = [(int(i), int(u), int(np.round(rat)))
                       for (i, u, _, rat, _) in df_predictions.values]
        header = 'Id,Prediction\n'
        data = [header]
        for pred in predictions:
            data.append('r{0}_c{1},{2}\n'.format(*pred))

        path = os.path.join('predictions/', modelname + '_' +
                            rmse.astype('str') + '.csv')
        fp = open(path, 'w')
        fp.writelines(data)
        fp.close()


def write_tests(modelname, rmse, df_predictions):
    '''Saves predictions in dataframe to file defined by filepath.'''
    predictions = [(int(i), int(u), int(np.round(rat)))
                   for (i, u, _, rat, _) in df_predictions.values]
    header = 'Id,Prediction\n'
    data = [header]
    for pred in predictions:
        data.append('r{0}_c{1},{2}\n'.format(*pred))

    path = os.path.join('tests/', modelname + '_' +
                        rmse.astype('str') + '.csv')
    fp = open(path, 'w')
    fp.writelines(data)
    fp.close()


def check_labels(modelname, rmse):
    print('Checking labels')
    original = open_file('data/data_train.csv')
    predicted = open_file('predictions/' + modelname + '_' + rmse.astype('str')
                          + '.csv')
    pred_labels = []
    for string in predicted:
        pred_labels.append(string.split(',')[0])
    ori_labels = []
    for string in original:
        ori_labels.append(string.split(',')[0])
    set(pred_labels) == set(ori_labels)

    if pred_labels == ori_labels:
        return True


def return_labels():
    data = open_file('data/data_train.csv')
    labels = open_file('data/data_test.csv')
    ratings = parse_as_dataset(data)
    labels = parse_as_dataset(labels)
    trainset, testset = train_test_split(ratings, test_size=0.2)

    algo = BaselineOnly()
    algo.fit(trainset)
    train_labels = algo.test(trainset.build_testset())
    test_labels = algo.test(testset)
    return train_labels, test_labels
