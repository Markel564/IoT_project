
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # Compilation of model
        self.model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mean_squared_error'])

        # Train the model
        self.model.fit(self.X_train_seq, self.y_train_seq, epochs=100, batch_size=32)
        


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


