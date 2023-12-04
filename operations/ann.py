
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tensorflow import keras
from keras.layers import Dropout


tf.get_logger().setLevel('ERROR')
class Ann:

    def __init__(self, y_test, y_train, window_size, output_size, target_variable):

        self.y_test = y_test
        self.y_train = y_train
        self.window_size = window_size
        self.output_size = output_size
        self.target_variable = target_variable

        self.build_model()
    
    def build_model(self):
        
        # Create sequences and labels for training data
        self.X_train_seq, self.y_train_seq = self.create_sequences_and_labels(self.y_train)

        # Create sequences and labels for test data
        self.X_test_seq, self.y_test_seq = self.create_sequences_and_labels(self.y_test)

        

        self.model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(self.window_size,), activation='relu'),
        Dropout(0.2),   # Dropout layer to prevent overfitting, getting rid of 20% of the neurons
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(self.output_size, activation='linear')
        ])

        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        # Compilation of model
        self.model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mean_squared_error'])

        # Train the model
        self.model.fit(self.X_train_seq, self.y_train_seq, epochs=50, batch_size=32)
        


    def create_sequences_and_labels(self, df):
        sequences, labels = [], []
        for i in range(len(df) - self.window_size - self.output_size + 1):
            seq = df[i:i + self.window_size]
            label = df[i + self.window_size:i +self.window_size + self.output_size]
            sequences.append(seq)
            labels.append(label)

        return np.array(sequences), np.array(labels)

    def evaluate(self):

        test_loss, test_mse = self.model.evaluate(self.X_test_seq, self.y_test_seq, verbose=2)

        rmse = np.sqrt(test_mse)
        return(np.round(rmse, 2))


    def predict(self):

        predictions = self.model.predict(self.X_test_seq[-1].reshape(1, self.window_size))
        return predictions[0]

    def custom_scorer(self, estimator, X, y):
        y_pred = estimator.predict(X)
        mse = np.mean((y - y_pred)**2)
        return -mse  # negative MSE because GridSearchCV looks for maximum score

        
    def perform_hyperparameter_tuning(self):
        # Define the grid search parameters
        param_grid = {
            'batch_size': [32, 64, 128],
            'epochs': [50, 100, 150],
            'optimizer': ['adam', 'sgd', 'rmsprop']
        }

        # Create TimeSeriesSplit cross-validator
        tscv = TimeSeriesSplit(n_splits=3)

        best_score = float('-inf')
        best_params = None

        # Perform grid search
        for batch_size in param_grid['batch_size']:
            for epochs in param_grid['epochs']:
                for optimizer in param_grid['optimizer']:
                    model = Sequential([
                        Dense(64, input_shape=(self.window_size,), activation='relu'),
                        Dropout(0.2),
                        Dense(32, activation='relu'),
                        Dense(16, activation='relu'),
                        Dense(self.output_size, activation='linear')
                    ])

                    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

                    # Track average score across time series splits
                    avg_score = 0.0

                    for train_index, val_index in tscv.split(self.X_train_seq):
                        X_train, X_val = self.X_train_seq[train_index], self.X_train_seq[val_index]
                        y_train, y_val = self.y_train_seq[train_index], self.y_train_seq[val_index]

                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                        val_score = self.custom_scorer(model, X_val, y_val)
                        avg_score += val_score

                    avg_score /= tscv.n_splits

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {'batch_size': batch_size, 'epochs': epochs, 'optimizer': optimizer}

        print("Best: %f using %s" % (best_score, best_params))
        return best_params

   
