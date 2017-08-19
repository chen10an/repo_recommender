# libraries
import numpy as np
from scipy.optimize import fmin_cg

# files
import utils

class User:
    def __init__(self, itemList):
        # initialize new user's ratings
        self.ratings = np.zeros(len(itemList))
        self.X = None
        self.Theta = None
        self.predictions = None
        self.ix = None
        self.top = None
        self.count = 0

    def printRatings(self, itemList):
        print("User ratings:")
        for i in range(len(self.ratings)):
            if self.ratings[i] > 0:
                print("Starred {}".format(itemList[i]))

    def train(self, Y, R, lambd=10, normalize=True, num_features=10, maxiter=100):
        print("\nTraining model...")

        # reshape ratings into a column vector
        ratings = np.reshape(self.ratings, (self.ratings.size,1))

        # add the new user's ratings to Y and R
        Y = np.concatenate((ratings,Y), axis=1)
        R = np.concatenate(((ratings != 0), R), axis=1)

        Y_in = Y
        if normalize:
            # normalize ratings
            Ynorm, Ymean = utils.normalizeRatings(Y,R)
            Y_in = Ynorm

        # useful values
        num_users = Y.shape[1]
        num_items = Y.shape[0]

        # set initial parameters (Theta, X)
        X = np.random.randn(num_items, num_features)
        Theta = np.random.randn(num_users, num_features)
        initial_parameters = np.concatenate((X,Theta))
        initial_parameters = np.reshape(initial_parameters, initial_parameters.size)

        # set functions for the cost and the gradient
        f = lambda t: utils.costFunc(t, Y_in, R, num_users, num_items, num_features, lambd)[0]
        fprime = lambda t: utils.costFunc(t, Y_in, R, num_users, num_items, num_features, lambd)[1]

        # minimize gradient with fmincg
        def callback(x):
            if self.count%10 == 0:
                print("iteration {}\tloss {}".format(self.count, f(x)))
            self.count += 1

        theta = fmin_cg(f=f, x0=initial_parameters, fprime=fprime,
        maxiter=maxiter, callback=callback)

        # unfold theta back into X and Theta
        self.X = np.reshape(theta[0:num_items*num_features], (num_items, num_features))
        self.Theta = np.reshape(theta[num_items*num_features:], (num_users, num_features))

    def predict(self, itemList, Y, R, n=10, normalize=True):
        assert self.X is not None
        assert self.Theta is not None

        # make predictions by multiplying X and Theta
        self.predictions = self.X.dot(self.Theta.T)[:,0]

        if normalize:
            # normalize ratings
            Ynorm, Ymean = utils.normalizeRatings(Y,R)
            self.predictions = self.predictions + Ymean

        assert n < len(self.predictions)

        # get the indices ix that would sort predictions in a descending order
        # by predictions[ix]
        self.ix = np.argsort(self.predictions)[::-1]

        self.top = []
        i = 0
        while i < n:
            j = self.ix[i]
            if self.ratings[j] == 0:  # only recommend unrated items
                self.top.append((np.rint(self.predictions[j]), itemList[j]))
                i += 1

    def printTop(self):
        assert self.top is not None

        print("\nTop {} recommendations:".format(len(self.top)))
        for i in range(len(self.top)):
            print("Predicting star for {}".format(self.top[i][1]))
