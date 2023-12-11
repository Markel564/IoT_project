import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


class LinearRegressionModel:
    def __init__(self, y_train, y_test, window_size, output_size, target_variable):
        self.y_train = y_train
        self.y_test = y_test
        self.window_size = window_size
        self.output_size = output_size
        self.target_variable = target_variable

        self.model = None
        self.best_model = None

    def build_model(self):
        # Define the base model
        self.model = LinearRegression()
        # Define the grid of hyperparameters to search
        param_grid = {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
            'n_jobs': [None, -1],
            'positive': [True, False]
        }
        # Create and fit the grid search
        grid_search = GridSearchCV(self.model, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(self.X_train_seq, self.y_train)
        # Store the best model
        self.best_model = grid_search.best_estimator_
        print("Best Hyperparameters:", grid_search.best_params_)

    def create_sequences_and_labels(self, df):
        # Implement the logic to create sequences and labels from the dataframe
        pass

    def evaluate(self):
        # Evaluate the model on the test set
        y_pred = self.best_model.predict(self.X_test_seq)
        mse = mean_squared_error(self.y_test, y_pred)
        print("Mean Squared Error on Test Set:", mse)
        return mse

    def predict(self):
        # Make predictions using the best model
        predictions = self.best_model.predict(self.X_test_seq)
        return predictions

# Usage example
model = LinearRegressionModel(y_train, y_test, 7, 3, target)
model.build_model()
mse = model.evaluate()
print(mse)
predictions = model.predict()
print(predictions)