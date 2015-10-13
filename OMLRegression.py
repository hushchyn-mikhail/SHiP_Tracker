__author__ = 'mikhail91'

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class OMLRegression(object):
    """
    This is the Orthogonal Margin Linear Regression in 2D.
    """

    def __init__(self):
        pass

    def fit(self,X,Y,R, sign, n=0):
        """
        Fit the regression.
        :param numpy.array X: The first coordinate of the points.
        :param numpy.array Y: The second coordinate of the points.
        :param numpy.array R: Margins between the line and the points.
        :param numpy.array sign: Signs of the margins.
        :param int n: NUmber of point that will be skipped during fit.
        :return: 1 in case of success.
        """
        X_new = np.concatenate((X,Y), axis=1)
        Y_new = R*sign
        sample = [np.random.choice(X_new.shape[0], X_new.shape[0] - n)]

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_new[sample], Y_new[sample])
        self.lr = lr
        coef = lr.coef_
        intersept = lr.intercept_

        #y = kx + b
        self.k = - float(coef[0][0])/float(coef[0][1])
        self.b = - float(intersept)/float(coef[0][1])
        return 1

    def predict(self,X):
        """
        Predict values of Y using values of X.
        :param numpy.array X: The first coordinate of the points.
        :return: numpy.array Y: The second coordinate of the points.
        """
        return self.k*X + self.b



class EMOMLRegression(object):
    """
    The EM-algorithm for the OMLRegression in case when the signs of R are unknown.

    :param int n_samp: number of samples that will be skipped din fit.
    :param int n_steps: number of step of the EM-algorithm.
    """

    def __init__(self, n_samp=1, n_steps=10):
        self.n_samp = n_samp
        self.n_steps = n_steps

    def fit(self, X, Y, R):
        """
        Fit the regression.
        :param numpy.array X: The first coordinate of the points.
        :param numpy.array Y: The second coordinate of the points.
        :param numpy.array R: Margins between the line and the points.
        :return: 1 in case of success.
        """
        X_new = X.reshape((X.shape[0], 1))
        Y_new = Y.reshape((Y.shape[0], 1))
        R_new = R.reshape((R.shape[0], 1))

        self.X_new = X_new
        self.Y_new = Y_new
        self.R_new = R_new

        self.regressors = []
        self.Rs = []

        #1-E step
        lr = LinearRegression(fit_intercept=True)
        sample = [np.random.choice(X_new.shape[0], X_new.shape[0] - self.n_samp)]
        lr.fit(X_new[sample],Y_new[sample])
        self.regressors.append(lr)
        self.Rs.append(R_new)

        #1-M step
        sign = 1.*(Y_new - lr.predict(X_new) > 0) - 1.*(Y_new - lr.predict(X_new) <= 0)

        #other steps
        for step in range(0, self.n_steps-1):
            #E step
            omlr = OMLRegression()
            omlr.fit(X_new,Y_new,R_new,sign)
            self.regressors.append(omlr)
            self.Rs.append(R_new*sign)
            #M step
            sign = 1.*(Y_new - omlr.predict(X_new) > 0) - 1.*(Y_new - omlr.predict(X_new) <= 0)

        self.regressors = np.array(self.regressors)
        self.Rs = np.array(self.Rs)
        return 1

    def _get_best_regressor(self):
        """
        This method finds the best regressor.
        :return: OMLRegression best regressor.
        """
        scores = self.get_train_learning_curve()
        best_regressor = self.regressors[scores == scores.min()][0]
        return best_regressor

    def predict(self, X):
        """
        Predict values of Y using values of X.
        :param numpy.array X: The first coordinate of the points.
        :return: numpy.array Y: The second coordinate of the points.
        """
        X_new = X.reshape((X.shape[0], 1))
        best_regressor = self._get_best_regressor()
        return best_regressor.predict(X_new)

    def _get_score(self, X, Y):
        """
        Get quality of the regressor.
        :param numpy.array X: The first coordinate of the points.
        :param numpy.array Y: The second coordinate of the points.
        :return: mean of (error)^2
        """
        return ((X - Y)**2).mean()

    def get_learning_curve(self, X, Y, R):
        """
        Get regressor qualities at each step of the EM-algorithms.
        :param numpy.array X: The first coordinate of the points.
        :param numpy.array Y: The second coordinate of the points.
        :param numpy.array R: Margins between the line and the points.
        :return: numpy.array means of (error)^2
        """
        X_new = X.reshape((X.shape[0], 1))
        Y_new = Y.reshape((Y.shape[0], 1))
        R_new = -R.reshape((R.shape[0], 1))

        scores = []

        for i in range(0, self.n_steps):
            Y_predict = self.regressors[i].predict(X_new)
            R_predict = Y_new - Y_predict
            score = self._get_score(R_predict, R_new)
            scores.append(score)
        return np.array(scores)

    def get_train_learning_curve(self):
        """
        Get regressor qualities at each step of the EM-algorithms using train data.
        :param numpy.array X: The first coordinate of the points.
        :param numpy.array Y: The second coordinate of the points.
        :param numpy.array R: Margins between the line and the points.
        :return: numpy.array means of (error)^2
        """
        scores = []

        for i in range(0, self.n_steps):
            Y_predict = self.regressors[i].predict(self.X_new)
            R_predict = self.Y_new - Y_predict
            score = self._get_score(R_predict, self.Rs[i])
            scores.append(score)
        return np.array(scores)
