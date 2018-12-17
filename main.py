from pandas import DataFrame
from surprise.model_selection import train_test_split
from data_processing import (open_file, parse_as_dataset, write_predictions)
from models import svd, user_knn
from helpers import set_random

# TODO: Experiment with how to increase the accuracy of low count hits.
# TODO: Extract predictions from SVD algorithm


# -------------------------      MAIN       ----------------------------------

# For reproducibility
set_random()

data = open_file('data/data_train.csv')
ratings = parse_as_dataset(data)

# TODO: separate input data into training and test data.
trainset, testset = train_test_split(ratings, test_size=0.2)

predictions, rmse = svd(trainset, testset, ratings, force=True)

write_predictions('SVD', rmse, DataFrame(predictions))

predictions, rmse = user_knn(trainset, testset, ratings)

write_predictions('userKNN', rmse, DataFrame(predictions))



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

import os
import glob
def already_written(modelname, rmse):
    '''Checks if predictions already exist.'''
    basepath = os.path.join('precitions')
    if glob(os.path.join(basepath, modelname + '_' + rmse.astype('str')
            + '.csv')):
        return True
    else:
        return False
