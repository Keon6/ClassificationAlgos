from numpy import dot, sign

def predict(w, x_test):
    """
    Returns the predictions for a weight vector and a data set.
    :param w:    (d x 1) weight vector (default w=0)
    :param x_test: (d x n) matrix (each column is an input vector)
    :return:     (1 x n) predictions for a weight vector and a data set.
    """
    predictions = sign(dot(w[:, 0], x_test))
    return predictions.reshape((1, len(predictions)))
