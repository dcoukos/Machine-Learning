from pandas import DataFrame
from surprise.model_selection import train_test_split
from data_processing import (open_file, parse_as_dataset, write_predictions)
from models import (svd, user_knn, movie_knn, baseline, slope_one,
                    co_clustering, return_labels)
from helpers import set_random

# TODO: Experiment with how to increase the accuracy of low count hits.
# TODO: Extract predictions from SVD algorithm


# -------------------------      MAIN       ----------------------------------

# For reproducibility
def main():
    set_random()
    # TODO: separate input data into training and test data.


    predictions, rmse = baseline()
    write_predictions('Baseline', rmse, DataFrame(predictions))
    predictions, rmse = slope_one()
    write_predictions('SlopeOne', rmse, DataFrame(predictions))

    #KNNs
    predictions, rmse = movie_knn()
    write_predictions('movieKNN', rmse, DataFrame(predictions))
    predictions, rmse = user_knn()
    write_predictions('userKNN', rmse, DataFrame(predictions))
    #Co-clustering
    predictions, rmse = co_clustering(10, 10)
    write_predictions('CoClustering', rmse, DataFrame(predictions))
    predictions1, rmse1 = co_clustering(5, 5)
    write_predictions('CoClustering', rmse1, DataFrame(predictions1))
    predictions2, rmse2 = co_clustering()
    write_predictions('CoClustering', rmse2, DataFrame(predictions2))

    predictions2, rmse2 = co_clustering(2,2)
    write_predictions('CoClustering_2factors', rmse2, DataFrame(predictions2))


if __name__ == '__main__':
    main()
