import numpy as np
"File containing helper functions developed during course exercises"


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''Linear regression using gradient descent.'''


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    '''Linear regression using stochastic gradient descent.'''
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size,
                                            num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def least_squares(y, tx):
    '''Least squares regression using normal equations'''
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    # solving for the inverse?
    return np.linalg.solve(a, b)


def ridge_regression(y, tx, lambda_):
    '''Ridge regression using normal equations'''
    a = tx.T.dot(tx) + 2 * tx.shape[0]*lambda_*np.eye(tx.shape[1])
    # Use tx's shape above, instead of len(y), because tx not a square matrix
    # tx.T.dot(tx) -> square, Gram matrix of size tx.shape[1]**2.
    b = tx.T.dot(y)
    return a*b


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''Logistic regression using gradient descent or SGD'''


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    '''Regularized logistic regression using gradient descent or SGD'''


def compute_gradient(y, tx, w):
    '''Computes gradient for gradient descent.

    Returns gradient for specific model paramter w when minimizing the losses
    of a convex loss function.

    Arguments:
    y --
    tx --
    w -- model parameters

    Returns:
    grad -- gradient
    err -- error between real and predicted observation
    '''
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the
    input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching
    elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data
    messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:
                                                                 end_index]
