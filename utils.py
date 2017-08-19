import numpy as np

# returns the cost and gradient (both with regularization)
# for the collaborative filtering problem
# params: 1D array containing feature matrix X and param vector Theta
# Y: num_items x num_users 2d array of user ratings of items
# R: num_items x num_users 2d array, where R[i, j] = 1 if the
# i-th item was rated by the j-th user
# num_users, num_items, num_features, lambd: scalars
def costFunc(params, Y, R, num_users, num_items, num_features, lambd):
    # X: num_items  x num_features 2d array of movie features
    # Theta: num_users  x num_features 2d array of user features
    X = np.reshape(params[0:num_items*num_features], (num_items, num_features));
    Theta = np.reshape(params[num_items*num_features:], (num_users, num_features));

    diff = ((Theta.dot(X.T)) * R.T) - (Y.T * R.T)
    J_reg = lambd/2 * np.sum(Theta**2) + lambd/2 * np.sum(X**2)
    X_grad_reg = lambd * X
    Theta_grad_reg = lambd * Theta

    # J: scalar cost value
    J = 1/2 * np.sum(diff**2) + J_reg

    # X_grad: num_items x num_features matrix, containing the
    # partial derivatives w.r.t. to each element of X
    # Theta_grad: num_users x num_features matrix, containing the
    # partial derivatives w.r.t. to each element of Theta
    X_grad = diff.T.dot(Theta) + X_grad_reg
    Theta_grad = diff.dot(X) + Theta_grad_reg

    grad = np.concatenate((X_grad, Theta_grad))
    grad = np.reshape(grad, grad.size)

    return J, grad

# Computes the gradient using "finite differences"
# and gives us a numerical estimate of the gradient.
# numgrad = gradientChecking(J, theta) computes the numerical
# gradient of the function J around theta.
# Calling y = J(theta) should return the function value at theta.
def gradientChecking(J, theta):
    # sets numgrad(i) to (a numerical approximation of)
    # the partial derivative of J with respect to the
    # i-th input argument, evaluated at theta. (i.e., numgrad(i) should
    # be the (approximately) the partial derivative of J with respect
    # to theta(i).)

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = e
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad

def checkCostFunction(lambd=0):
    # creates a collaborative filtering problem to check cost function and gradients
    # outputs the analytical gradients produced by costFunc
    # and the numerical gradients (computed using gradientChecking)
    # these two gradient computations should result in very similar values

    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = X_t.dot(Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_items = Y.shape[0]
    num_features = Theta_t.shape[1]

    params = np.concatenate((X,Theta))
    params = np.reshape(params, params.size)

    numgrad = gradientChecking(lambda t: costFunc(t, Y, R, num_users, num_items,
    num_features, lambd), params)

    cost, grad = costFunc(params, Y, R, num_users, num_items, num_features, lambd)

    numgrad = np.reshape(numgrad, (numgrad.size, 1))
    grad = np.reshape(grad, (grad.size, 1))
    print(np.concatenate((numgrad, grad), axis=1))
    print('The above two columns should be very similar.\n'\
    '(Left-Numerical Gradient, Right-Analytical Gradient)\n\n')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If the cost function implementation is correct, then \n'\
    'the relative difference will be small (less than 1e-9). \n'\
    '\nRelative Difference: {}\n'.format(diff))

def normalizeRatings(Y, R):
# Preprocess data by subtracting mean rating for every movie (every row)
# normalizes Y so that each movie has a rating of 0 on average,
# and returns the mean rating in Ymean

    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.where(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean
