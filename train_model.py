import numpy as np
from loss_functions import ridge, hinge, logistic
from linear_model_prediction import predict


def train_model(func_type, x_tr, y_tr):
    """
    Trains your spam filter & saves the final weight vector in a file w_trained.mat
    :params func: str- loss function to use
    :param x_tr: (d x n) matrix (each column is an input vector)
    :param y_tr: (1 x n) matrix (each entry is a label)
    :return:     (d x 1) w_trained = final trained weight vector
    """
    # IDEA:
    # k-fold CV to find a good regularizing parameter lambda
    
    # Logistic has no regularizing parameter
    if func_type is "logistic":
        f = lambda w: logistic(w, x_tr, y_tr)
        return grdescent(f, np.zeros((x_tr.shape[0], 1)), 1e-01, 2000)
    
    
    # Hinge and Ridge has regularizing parameters
    # 0) Preparation : Pre-do Computation for Commonly used data
    #  ->   k-fold CV information
    n, k = x_tr.shape[1], 5
    val_size = n // k
    # ->    Partition Data for k-fold CV
    cv_indices, train_indices = [], []
    for i in range(k):
        # Find indices to separate CV and Training sets
        cv_indices.append((i * val_size, (i + 1) * val_size))  # (start,end)
        train_indices.append((0, i * val_size, (i + 1) * val_size, n))  # (start1, end1, start2, end2)

    # 1) Use k-CV to find appropriate lambda for the regularizer
    best_lambda = [0, float("inf")]  # (lambda, E_cv)
    lambdas = [0, 1e-5, 0.0001, 0.0025, 0.0035, 0.0045, 0.005, 0.0055, 0.0065, 0.0075, 0.0095, 0.01,
               0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 5]
    for lamb in lambdas:
        e_cv = 0
        for i in range(k):
            # Separate CV and Training sets
            x_cv, y_cv = x_tr[:, cv_indices[i][0]:cv_indices[i][0]], y_tr[:, cv_indices[i][0]:cv_indices[i][0]]
            x_train = np.column_stack((x_tr[:, train_indices[i][0]:train_indices[i][1]], x_tr[:, train_indices[i][2]:train_indices[i][3]]))
            y_train = np.column_stack((y_tr[:, train_indices[i][0]:train_indices[i][1]], y_tr[:, train_indices[i][2]:train_indices[i][3]]))

            if func_type is "ridge":
                # Matrix computation
                A = np.linalg.inv(np.dot(x_train, x_train.T) + lamb * np.identity(n=x_train.shape[0]))
                B = np.dot(x_train, y_train.T)
                w_cv = np.dot(A, B)
                del A, B
            
            elif func_type is "hinge":
                f = lambda w: hinge(w, x_train, y_train, lamb)
                w_cv = grdescent(f, np.zeros((x_train.shape[0], 1)), 1e-01, 2000)
            
            predictions = predict(w_cv, x_cv)
            e_cv += np.sum(np.multiply(y_cv[0] != predictions[0], 1)) / len(predictions[0])
            # For optimization, if cross E_cv is already greater than other lambda, break
            if e_cv > best_lambda[1]:
                break
        # update lambda with lowest e_cv
        if e_cv < best_lambda[1]:
            best_lambda = [lamb, e_cv]

    # 2) Train Final Model
    if func_type is "ridge":
        # Matrix computation
        A = np.linalg.inv(np.dot(x_tr, x_tr.T) + best_lambda[0] * np.identity(n=x_tr.shape[0]))
        B = np.dot(x_tr, y_tr.T)
        return np.dot(A, B)
            
    elif func_type is "hinge":
        f = lambda w: hinge(w, x_tr, y_tr, best_lambda[0])
        return grdescent(f, np.zeros((x_tr.shape[0], 1)), 1e-01, 2000)
