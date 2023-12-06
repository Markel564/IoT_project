
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tensorflow import keras
from keras.layers import Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


tf.get_logger().setLevel('ERROR')
class Ann:

    def __init__(self, df, target_variable):
        
        self.df = df

        self.window_size = 7
        self.output_size = 3
        self.target_variable = target_variable

        self.x_conversion()
        self.y_conversion()

        self.split_data(0.7)

        self.build_model()
    

    def x_conversion(self):
        sequences = []
        for i in range(len(self.df) - self.window_size - self.output_size): 
            seq = self.df.drop(self.target_variable, axis=1).iloc[i : i + self.window_size]

            sequences.append(seq)

        # now, lets create X_train and X_test
        self.X_train = np.array(sequences[:int(len(sequences) * 0.7)])
        self.X_test = np.array(sequences[int(len(sequences) * 0.7):])
        
        

    def y_conversion(self):

        labels = []
        for i in range(self.window_size, len(self.df) - self.output_size): 
            label = self.df[self.target_variable].iloc[i : i + self.output_size]
            labels.append(label)
        
        # now, lets create y_train and y_test
        self.y_train = np.array(labels[:int(len(labels) * 0.7)])
        self.y_train = self.y_train.reshape(self.y_train.shape[0], self.output_size)
        self.y_test = np.array(labels[int(len(labels) * 0.7):])
        self.y_train = self.y_train.reshape(self.y_train.shape[0], self.output_size)
        

    def split_data(self, train_size):
        # Now we will split the windowed data into train and test sets
        # And shuffle the data as order is no longer important as the data is windowed

        target_variables = ['temperature_celsius', 'wind_kph', 'humidity', 'pressure_mb', 'precip_mm', 'cloud']

        if self.target_variable not in target_variables:
            raise ValueError(f"Target variable must be one of {target_variables}")

        # we already have X_train and X_test and y_train and y_test, but we can shuffle the windows for better 

        # TO DO: Shuffle the windows
        indices = np.arange(len(self.X_train))
        np.random.shuffle(indices)

        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]


    def build_model(self):
        self.model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(self.window_size, self.X_train.shape[2]), activation='relu'),
        Dropout(0.2),   # Dropout layer to prevent overfitting, getting rid of 20% of the neurons
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(self.output_size, activation='linear'),
        # keras.layers.TimeDistributed(keras.layers.Dense(self.output_size, activation='linear')),  # Output for each time step

        ])

        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        # Compilation of model
        self.model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mean_squared_error'])


        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, verbose=1)


    def evaluate(self):

        test_loss, test_mse = self.model.evaluate(self.X_test, self.y_test, verbose=2)

        rmse = np.sqrt(test_mse)
        return rmse

    def predict(self, df):
        # Assuming self.df was used during training, use the same scaler

        X = df.drop(self.target_variable, axis=1)
        X = X.iloc[-self.window_size:]

        X = np.array(X)

        X = X.reshape(1, X.shape[0], X.shape[1])
        predictions = self.model.predict(X)

        return predictions[0]

        

#    indices = np.arange(len(X_train))
#         np.random.shuffle(indices)

#         X_train = X[indices[:train_size]]
#         y_train = y[indices[:train_size]]
#         X_test = 

#         self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

#         val_score = self.custom_scorer(self.model, X_val, y_val)
#         avg_score += val_score

#     avg_score /= tscv.n_splits

#     if avg_score > best_score:
#         best_score = avg_score
#         best_params = {'batch_size': batch_size, 'epochs': epochs, 'optimizer': optimizer}