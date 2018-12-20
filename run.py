from pandas import DataFrame
import pandas as pd
import numpy as np
from data_processing import (open_file, rearrange_columns, parse_as_dataframe, parse_as_dataset, parse_as_dataset_from_df, preprocess, write_predictions, write_tests, write_training_as_test)
from models import (svd, user_knn, movie_knn, baseline, slope_one, co_clustering)
from helpers import set_random
import random
from surprise import Dataset
from surprise.model_selection import train_test_split



# For reproducibility

def main():

    #---------------------SET RANDOM SEED -------------------------
    random.seed(0)
    np.random.seed(0)

    #--------------------- DEFINE DATASETS -------------------------
    the_data = pd.read_csv('data/data_train.csv', header=0, index_col=0, names=['Id', 'rating'])

    test_dimitri = pd.read_csv('predictions/Testing_labels_0.9993.csv', header=0, index_col=0, names=['Id', 'rating'])
    test_indices = test_dimitri['rating'].index.values

    train_dimitri = pd.read_csv('predictions/Training_labels_0.9993.csv', header=0, index_col=0, names=['Id', 'rating'])
    train_indices = train_dimitri['rating'].index.values

    pre_test = the_data.loc[test_indices]
    pre_train = the_data.loc[train_indices]

    test = preprocess(pre_test)
    train = preprocess(pre_train)
    cols = test.columns.tolist()
    cols = [cols[2], cols[1], cols[0]]
    test = test[cols]
    train = train[cols]
    train
    trainset = parse_as_dataset_from_df(train)
    testset = parse_as_dataset_from_df(test)
    trainset = trainset.build_full_trainset()
    testset = testset.build_full_trainset().build_testset()
    data = open_file('data/data_train.csv')
    labels = open_file('data/data_test.csv')
    labels = parse_as_dataset(labels)
    data = parse_as_dataset(data)
    fullset = data.build_full_trainset()

    # --------------------BASELINE -------------------------------
    predictions, rmse = baseline(trainset, testset, fullset, labels, train_as_test=True)
    write_training_as_test('Baseline', rmse, DataFrame(predictions))

    predictions, rmse = baseline(trainset, testset, fullset, labels, testing=True)
    write_tests('Baseline', rmse, DataFrame(predictions))

    predictions, rmse = baseline(trainset, testset, fullset, labels)
    write_predictions('Baseline', rmse, DataFrame(predictions))
    ## --------------------SLOPE ONE -------------------------------
    predictions, rmse = slope_one(trainset, testset, fullset, labels, train_as_test=True)
    write_training_as_test('SlopeOne', rmse, DataFrame(predictions))

    predictions, rmse = slope_one(trainset, testset, fullset, labels, testing=True)
    write_tests('SlopeOne', rmse, DataFrame(predictions))

    predictions, rmse = slope_one(trainset, testset, fullset, labels)
    write_predictions('SlopeOne', rmse, DataFrame(predictions))

    ## --------------------Co-Cluster 2 Factors -------------------------------
    predictions, rmse2 = co_clustering(trainset, testset, fullset, labels, 2, 2, train_as_test=True)
    write_training_as_test('CoClustering_2factors', rmse, DataFrame(predictions))

    predictions, rmse2 = co_clustering(trainset, testset, fullset, labels, 2, 2, testing=True)
    write_tests('CoClustering_2factors', rmse, DataFrame(predictions))

    predictions, rmse = co_clustering(trainset, testset, fullset, labels, 2, 2)
    write_predictions('CoClustering_2factors', rmse, DataFrame(predictions))
    # ------------------- SVD ---------------------------------
    predictions, rmse = svd(trainset, testset, fullset, labels,
                            n_factors=2, train_as_test=True)
    write_training_as_test('SVD_2factors', rmse, DataFrame(predictions))

    predictions, rmse = svd(trainset, testset, fullset, labels,
                            n_factors=2, testing=True)
    write_tests('SVD_2factors', rmse, DataFrame(predictions))

    predictions, rmse = svd(trainset, testset, fullset, labels,
                            n_factors=2)
    write_predictions('SVD_2factors', rmse, DataFrame(predictions))




if __name__ == '__main__':
    main()
