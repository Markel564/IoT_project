import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from .adjust_dataset import adjust_dataset

class LinearRegressionModel:
    def __init__(self, y_train, y_test, window_size, output_size, target_variable):
        self.y_train = y_train
        self.y_test = y_test
        self.window_size = window_size
        self.output_size = output_size
        self.target_variable = target_variable
        self.model = None

    def build_model(self):
        self.model = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, positive=True)

    def train_model(self, sample_weight=None):
        X_train, y_train = self.create_sequences(self.y_train)
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self):
        X_test, y_test = self.create_sequences(self.y_test)
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return mse, rmse

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size - self.output_size + 1):
            X.append(data[i:(i + self.window_size)])
            y.append(data[i + self.window_size:i + self.window_size + self.output_size])
        return np.array(X), np.array(y)

