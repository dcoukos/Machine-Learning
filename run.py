from pandas import DataFrame
from data_processing import (write_predictions, write_tests)
from models import (svd, user_knn, movie_knn, baseline, slope_one, co_clustering)
from helpers import set_random


# -------------------------      Baseline, Slope1, MovieKNN       ------------

# For reproducibility
def main():
    set_random()
    # Generate test predictions for blending coefficients.
    predictions, rmse = baseline()
    write_tests('Baseline', rmse, DataFrame(predictions))
    #SlopeOne
    predictions, rmse = slope_one()
    write_tests('SlopeOne', rmse, DataFrame(predictions))
    #KNNs
    predictions, rmse = movie_knn()
    write_tests('movieKNN', rmse, DataFrame(predictions))
    predictions, rmse = user_knn()
    write_tests('userKNN', rmse, DataFrame(predictions))
    #Co-clustering
    predictions, rmse = co_clustering(10, 10)
    write_tests('CoClustering_10factors', rmse, DataFrame(predictions))
    predictions, rmse2 = co_clustering(2, 2)
    write_tests('CoClustering_2factors', rmse, DataFrame(predictions))
    #SVD
    predictions, rmse = svd(n_factors=2)
    write_tests('SVD_2factors', rmse, DataFrame(predictions))
    predictions, rmse = svd(n_factors=400)
    write_tests('SVD_400factors', rmse, DataFrame(predictions))

    # Generate predictions for CrowdAI
    #Baseline
    predictions, rmse = baseline()
    write_predictions('Baseline', rmse, DataFrame(predictions))

    #SlopeOne
    predictions, rmse = slope_one()
    write_predictions('SlopeOne', rmse, DataFrame(predictions))

    #KNNs
    predictions, rmse = movie_knn()
    write_predictions('movieKNN', rmse, DataFrame(predictions))
    predictions, rmse = user_knn()
    write_predictions('userKNN', rmse, DataFrame(predictions))

    #Co-clustering
    predictions, rmse = co_clustering(10, 10)
    write_predictions('CoClustering_10factors', rmse, DataFrame(predictions))
    predictions, rmse = co_clustering(2, 2)
    write_predictions('CoClustering_2factors', rmse, DataFrame(predictions))

    #SVD
    predictions, rmse = svd(n_factors=2)
    write_predictions('SVD_2factors', rmse, DataFrame(predictions))
    predictions, rmse = svd(n_factors=400)
    write_predictions('SVD_400factors', rmse, DataFrame(predictions))



if __name__ == '__main__':
    main()
