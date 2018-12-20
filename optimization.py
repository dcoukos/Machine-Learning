'''This file contains scripts that can be parallelized.'''

from multiprocessing import Pool
from models import user_knn, svd
from pandas import DataFrame
from surprise.model_selection import train_test_split
from data_processing import (open_file, parse_as_dataset, write_predictions)


def optimize_svd():
    factors = 5
    epoch = 20
    learning = 0.05
    reg = 0.002
    # Initial testing already indicates that 5 factors perform better,
    # under above conditions, therefore we skip this in first optimization
    epoch, learning, reg = find_svd_parameters(epoch=epoch,
                                               learning=learning, reg=reg, first=False,
                                               start_with_reg=True)

    factors, epoch, learning, reg = find_svd_parameters(epoch=epoch,
                                                        learning=learning,
                                                        reg=reg, first=False)
    data = open_file('data/data_train.csv')
    ratings = parse_as_dataset(data)
    test = open_file('data/data_test.csv')
    test = parse_as_dataset(test)

    # TODO: separate input data into training and test data.
    trainset, testset = train_test_split(ratings, test_size=0.2)
    predictions, rmse = svd(trainset, testset, test, n_factors=factors,
                            n_epochs=epoch, lr_all=learning,
                            reg_all=reg, force=True)
    write_prediction('SVD', rmse, np.DataFrame(predictions))


def find_svd_parameters(epoch=20, learning=0.005, reg=0.002, best_factor=5, first=True, start_with_reg=False):
    '''Defaults:
        factors = 100
        epochs = 20
        lr_all = 0.005
        reg_all = 0.02
    '''
    data = open_file('data/data_train.csv')
    ratings = parse_as_dataset(data)
    test = open_file('data/data_test.csv')
    test = parse_as_dataset(test)

    # TODO: separate input data into training and test data.
    trainset, testset = train_test_split(ratings, test_size=0.2)
    if not start_with_reg:
        if first is True:
            parameters = []
            n_factors = (2**exp for exp in range(1, 6))
            for f in n_factors:
                parameters.append((f, epoch, learning, reg, trainset, testset, test))
            best_factor = parameters[svd_multiprocess(parameters)[0]][0]
            print('Best factor: ', best_factor)
        parameters = []
        n_epochs = range(20, 120, 20)
        for ep in n_epochs:
            parameters.append((best_factor, ep, learning, reg, trainset,
                               testset, test))
        best_epoch = parameters[svd_multiprocess(parameters)[0]][1]
        print('Best epoch: ', best_epoch)
        parameters = []
        # Must have at least 4 factors
        lr_all = (0.0005*10**exp for exp in range(1, 5))
        for lr in lr_all:
            parameters.append((best_factor, best_epoch, lr, reg, trainset,
                               testset, test))
        best_learning_rate = parameters[svd_multiprocess(parameters)[0]][2]
        print('Best learning rate: ', best_learning_rate)
    if start_with_reg:
        best_factor = factors
        best_epoch = epoch
        best_learning_rate = learning
    parameters = []
    reg_all = (0.02*f for f in range(0, 50, 10))
    for reg in reg_all:
        parameters.append((best_factor, best_epoch, best_learning_rate, reg,
                           trainset, testset, ratings))
    best_regularization = parameters[svd_multiprocess(parameters)[0]][3]
    print('Best regularization: ', best_regularization)
    return best_factor, best_epoch, best_learning_rate, best_regularization


def svd_multiprocess(parameters):
    sets = distribute_parameters(parameters)
    set_sizes = (len(sets[0]), len(sets[1]), len(sets[2]), len(sets[3]))
    output1 = []
    output2 = []
    output3 = []
    output4 = []
    outputs = []
    '''Test with svd model first...'''
    # TODO: create trainint and testing sets
    # TODO: format parameters to be compatible with model
    # TODO: select return value, and pass to next loop
    with Pool(processes=4) as pool:
        # Use different output lists to prevent data mixing.
        output1 = pool.apply_async(svd_multiprocess_sub, (sets[0],))
        output2 = pool.apply_async(svd_multiprocess_sub, (sets[1],))
        output3 = pool.apply_async(svd_multiprocess_sub, (sets[2],))
        output4 = pool.apply_async(svd_multiprocess_sub, (sets[3],))
        pool.close()
        pool.join()
    outputs += list(output1.get())
    outputs += list(output2.get())
    outputs += list(output3.get())
    outputs += list(output4.get())
    return best_rmse_index(outputs, set_sizes)


def svd_multiprocess_sub(parameters):
    '''Function called by pool worker. Runs recommender model.'''
    output = []
    for param_set in parameters:
        output.append(svd(trainset=param_set[4], testset=param_set[5],
                          fullset=param_set[6],
                          n_factors=param_set[0], n_epochs=param_set[1],
                          lr_all=param_set[2], reg_all=param_set[3],
                          testing=True))
    return output


def best_rmse_index(model_output, set_sizes):
    min = float('inf')
    best_index = -1
    for index, out in enumerate(model_output):
        tr_rmse = out[3]
        test_rmse = out[2]
        delta_rmse = test_rmse - tr_rmse
        if delta_rmse < min and test_rmse < 1.1:
            min = delta_rmse
            best_index = index
    best_original = calculate_original_index(best_index, set_sizes)
    algo = model_output[best_index][0]
    predictions = model_output[best_index][1]
    return best_original, algo, predictions


def calculate_original_index(index, sizes):
    # Calulculation of size of lists passed to worker pools
    # TODO: correct calculation
    if index < sizes[0]-1:
        return index*4
    elif index < sizes[0] + sizes[1] - 1:
        index = index - sizes[0]
        return 1 + index*4
    elif index < sizes[0] + sizes[1] + sizes[2] - 1:
        index = index - sizes[0] - sizes[1]
        return 2 + index*4
    else:
        index = index - sizes[0] - sizes[1] - sizes[2]
        return 3 + index*4


'''
def find_knn_parameters():
    parameters = []
    k = range(5, 100, 5)
    min_support = range(0, 18, 3)
    for k_ in k:
        for ms in min_support:
            parameters.append((k_, ms))
    parameters
    set1, set2, set3, set4 = distribute_parameters(parameters)
    output1 = []
    output2 = []
    output3 = []
    output4 = []
    """Test with svd model first..."""
    with Pool(processes=4) as pool:
        for param_set:
            output1 = pool.apply_async(svd, (*))'''


def knn_user_multiprocess(parameters):
    sets = distribute_parameters(parameters)
    set_sizes = (len(sets[0]), len(sets[1]), len(sets[2]), len(sets[3]))
    output1 = []
    output2 = []
    output3 = []
    output4 = []
    outputs = []
    '''Test with svd model first...'''
    # TODO: create trainint and testing sets
    # TODO: format parameters to be compatible with model
    # TODO: select return value, and pass to next loop
    with Pool(processes=4) as pool:
        # Use different output lists to prevent data mixing.
        output1 = pool.apply_async(knn_user_multiprocess_sub, (sets[0],))
        output2 = pool.apply_async(knn_user_multiprocess_sub, (sets[1],))
        output3 = pool.apply_async(knn_user_multiprocess_sub, (sets[2],))
        output4 = pool.apply_async(knn_user_multiprocess_sub, (sets[3],))
        pool.close()
        pool.join()
    outputs += list(output1.get())
    outputs += list(output2.get())
    outputs += list(output3.get())
    outputs += list(output4.get())
    return best_rmse_index(outputs, set_sizes)


def knn_user_multiprocess_sub(parameters):
    '''Function called by pool worker. Runs recommender model.'''
    output = []
    for param_set in parameters:
        output.append(userKNN(trainset=param_set[4], testset=param_set[5],
                              fullset=param_set[6], range=param_set[0]
                              min_support=param_set[1], algorithm='basic',
                              force=False))
    return output


def distribute_parameters(parameters):
    set1 = []
    set2 = []
    set3 = []
    set4 = []
    for n, param_set in enumerate(parameters):
        if n % 4 == 0:
            set1.append(param_set)
        elif n % 4 == 1:
            set2.append(param_set)
        elif n % 4 == 2:
            set3.append(param_set)
        elif n % 4 == 3:
            set4.append(param_set)
    return [set1, set2, set3, set4]
