from numpy import dot, maximum, sum, exp, log


def ridge(w, x_tr, y_tr, lambdaa):
    """
    Computes the ridge regression loss and gradient.
    :param w:       (d x 1) weight vector (default w=0)
    :param x_tr:    (d x n) matrix (each column is an input vector)
    :param y_tr:    (1 x n) matrix (each entry is a label)
    :param lambdaa: (scalar) regression constant
    :return:        (scalar) loss = the total loss obtained with w on x_tr and y_tr &
                    (d x 1) gradient = the gradient of loss at w
    """

    w = w[:, 0]
    p = (dot(w, x_tr) - y_tr)[0]
    loss = dot(p, p) + lambdaa*dot(w, w)

    gradient = 2 * (dot(x_tr, p) + lambdaa * w)
    gradient = gradient.reshape(len(gradient), 1)

    return loss, gradient


def hinge(w, x_tr, y_tr, lambdaa):
    """
    Computes the ridge regression loss and gradient.
    :param w:       (d x 1) weight vector (default w=0)
    :param x_tr:    (d x n) matrix (each column is an input vector)
    :param y_tr:    (1 x n) matrix (each entry is a label)
    :param lambdaa: (scalar) regression constant
    :return:        (scalar) loss = the total loss obtained with w on x_tr and y_tr &
                    (d x 1) gradient = the gradient of loss at w
    """

    w = w[:, 0]
    hinges = (1 - y_tr * (dot(w, x_tr)))[0]
    loss = sum(maximum(0, hinges)) + lambdaa*dot(w, w)

    gradient = -dot(x_tr, (hinges > 0) * y_tr[0]) + 2 * lambdaa * w
    gradient = gradient.reshape(len(gradient), 1)

    return loss, gradient


def logistic(w, x_tr, y_tr):
    """
    Computes the logistic regression loss and gradient.
    :param w:    (d x 1) weight vector (default w=0)
    :param x_tr: (d x n) matrix (each column is an input vector)
    :param y_tr: (1 x n) matrix (each entry is a label)
    :return:     (scalar) loss = the total loss obtained with w on x_tr and y_tr &
                 (d x 1) gradient = the gradient of loss at w
    """
    inside = y_tr[0] * dot(w[:, 0], x_tr)
    loss = sum(log(1 + exp(-inside)))

    gradient = dot(x_tr, -y_tr[0]/(1 + exp(inside)))
    gradient = gradient.reshape(len(gradient), 1)
    return loss, gradient
