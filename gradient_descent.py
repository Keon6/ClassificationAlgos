import numpy as np
from numpy.linalg import norm


def gradient_descent(func, w_initial, step_size, max_iter, tolerance=1e-02):
    """
    Gradient Descent.
    :param func:      (2-tuple: (scalar), (1 x d)) function to minimize, gradient
    :param w_initial:        (d x 1) initial weight vector
    :param step_size: (scalar) initial gradient descent step size
    :param max_iter:  (scalar) maximum number of iterations
    :param tolerance: (scalar) if norm(gradient)<tolerance, it quits
    :return:          (d x 1) w = final weight vector
    """
    # Quasi-Newton Method

    # Updated Info
    t = 0  # iter0 -> iter1 -> ... ->
    B = np.identity(n=len(w_initial))  # (d x d) Inverse Hessian Approximation
    w = w_initial

    # Fixed Info
    eps = 2.2204e-14  # minimum step size for gradient descent
    I = np.identity(n=len(w_initial))  # Identity Matrix

    # GD-QN
    loss, gradient = func(w_initial)  # initial loss, gradient
    while norm(gradient) > tolerance and t < max_iter:

        #####
        # if t % 10 == 0:
        #    print("t: ", t, " | loss: ", loss)
        ####

        # Descent Vector
        s = -step_size*np.dot(B, gradient)  # (d x 1)
        # Update Weight
        w_update = w + s  # (d x 1)
        # updated loss, gradient
        loss_update, gradient_update = func(w_update)
        # update step_size
        if loss_update >= loss:
            step_size = step_size * 0.5
            w_update = 1*w
        else:
            step_size = 1.01 * step_size

        # update hessian approximation
        grad_diff = gradient_update - gradient  # (d x 1)
        # update Inverse Hessian Approximation
        d = 1 / np.dot(s[:, 0], grad_diff[:, 0])
        inter = d * np.dot(s, grad_diff.reshape(1, grad_diff.shape[0]))
        inter_tr = d * np.dot(grad_diff, s.reshape(1, s.shape[0]))
        if t == 0:
            B = ((np.dot(s[:, 0], grad_diff[:, 0]))/(np.dot(grad_diff[:, 0], grad_diff[:, 0]))) * B
        # (d x d) (d x d) (d x d)
        B = np.dot(np.dot(I - inter, B), I - inter_tr) + d*np.dot(s, s.reshape(1, s.shape[0]))

        # update time & weight & others
        t += 1
        w = w_update

        # if step_size > eps:
        #     step_size = step_size*((tao+t)/(tao+t+1))
        if step_size < eps:
            step_size = eps
        loss = 1*loss_update
        gradient = 1*gradient_update

    return w



