"""
-- This file contains the implementation of the ANN model --

A description of each function is provided in the docstring of each function.

A sliding window is used as an approach. The window size is 7 days and the output size is 3 days.
The model is trained with 70% of the data and tested with the remaining 30%.

"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid

torch.manual_seed(42)

class Ann(nn.Module):
    def __init__(self, df, target_variable):
        super(Ann, self).__init__()
        """
        Constructor of the ANN class
        """

        self.df = df # receives a full dataframe standarized and adjusted (see adjust_dataset.py)
        self.window_size = 7 # 7 days predicts 3 days
        self.output_size = 3
        self.target_variable = target_variable
        

        self.x_conversion(0.7)  # 70% of the data is used for training
        self.y_conversion(0.7)  # 70% of the data is used for training

        self.shuffle_data() # shuffle the data (the windows, so order is maintained)
        self.build_model()  # build the model as soon as it is instantiated

    def x_conversion(self, train_size):
        """
        input: train_size (float)
        output: None

        This function creates the windows of the input data and converts it into a tensor of shape (num_samples, window_size, num_features)
        It creates what we could consider as the X_train and X_test of the ANN
        """
        sequences = []
        # Iterate over each city (we want thw windows to be of the same city)
        for city in self.df['location_name'].unique():
            # we only get the data from each city
            city_data = self.df[self.df['location_name'] == city]
            # and create the windows for each city
            for i in range(len(city_data) - self.window_size - self.output_size + 1):   
                # drop the target variable (as this is X) and the location name since it is not necessary for prediction
                seq = city_data.iloc[i: i + self.window_size].drop([self.target_variable, 'location_name'], axis=1) 

                sequences.append(seq.values)  

        # transform the list into a tensor
        self.X_train = torch.tensor(sequences[:int(len(sequences) * train_size)], dtype=torch.float32)
        self.X_test = torch.tensor(sequences[int(len(sequences) * train_size):], dtype=torch.float32)

    
    def save_model(self, filepath): 
        """
        Saves the model's state dictionary to the specified filepath.
        """
        torch.save(self.state_dict(), filepath)    

    def y_conversion(self, train_size):
        """
        input: train_size (float)
        output: None

        This function creates the windows of the input data and converts it into a tensor of shape (num_samples, window_size, num_features)
        It creates what we could consider as the y_train and y_test of the ANN
        """
        labels = []
        for city in self.df['location_name'].unique(): # Iterate over each city (again, we want the windows to be of the same city)
            # we select the data of each city
            city_data = self.df[self.df['location_name'] == city]

            for i in range(self.window_size, len(city_data) - self.output_size + 1):
                # and create a window with the target variable
                label = city_data.iloc[i: i + self.output_size][self.target_variable]
                labels.append(label.values)
        
        # transform the list into a tensor
        self.y_train = torch.tensor(labels[:int(len(labels) * train_size)], dtype=torch.float32)
        self.y_test = torch.tensor(labels[int(len(labels) * train_size):], dtype=torch.float32)


        
    def shuffle_data(self):
        """
        input: None
        output: None

        This function shuffles the data for a better prediction
        """
        indices = torch.randperm(len(self.X_train))
        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]

    def build_model(self):
        """
        input: None
        output: None

        This function builds the model of the ANN, defining the layers and the optimizer
        """
        self.model = nn.Sequential(
            nn.Linear(self.window_size * self.X_train.size(2), 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.output_size)
        )

        # the optimizer is an Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # we use MSE as the loss function
        self.criterion = nn.MSELoss()
        
    
    def train_model(self, epochs=50, batch_size=32,  max_grad_norm=2.0):
        """
        input: epochs (int), batch_size (int), max_grad_norm (float)
        output: None

        This function trains the model with the specified hyperparameters
        """
        # we create a dataset and a dataloader for the training
        dataset = TensorDataset(self.X_train.view(-1, self.window_size * self.X_train.size(2)), self.y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # we create a list to store the training losses
        train_losses = []

        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                # we set the gradients to zero
                self.optimizer.zero_grad()
                # the predictions are made
                predictions = self(X_batch)
                # and the loss is computed
                loss = self.criterion(predictions, y_batch)
                # the gradients are computed
                loss.backward()
                # and the gradients are clipped
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                # the optimizer takes a step
                self.optimizer.step()
                # we append the loss to the list
                train_losses.append(loss.item())



    def evaluate(self):
        """
        input: None
        output: rmse (float)

        This function evaluates the model with the test data
        """
        with torch.no_grad():
            # we make the predictions
            predictions = self(self.X_test.view(-1, self.window_size * self.X_test.size(2)))

            # and compute the rmse
            test_loss = self.criterion(predictions, self.y_test)
            rmse = torch.sqrt(test_loss)
        
        # return the rmse calculated
        return rmse.item()

    
    
    def predict(self, df):
        """
        input: df (dataframe) of 7 days of the same city
        output: predictions (array) of 3 days, in the format array([day1, day2, day3])

        This function predicts the next 3 days of the input dataframe
        """
        # as we receive a dataframe with the 7 days containing the target variable, we drop it
        X = df.drop(self.target_variable, axis=1).iloc[-self.window_size:].values
        # and convert it into a tensor
        X = torch.tensor(X, dtype=torch.float32).reshape(1, -1, self.window_size * self.X_train.size(2))
        with torch.no_grad():
            # a prediction is made
            predictions = self(X)
       
        # return the 3 days predicted (predictions is an array of arrays, so we select the first one; it is how
        # the model is implemented in PyTorch)
        return predictions[0].numpy()


    def forward(self, x):
        """
        input: x (tensor)
        output: self.model(x)

        This function returns the model with the input tensor
        """
        return self.model(x)


    def hyperparameter_tuning(self, param_grid, num_epochs_list=[10, 20]):
        """
        input: param_grid (dictionary), num_epochs_list (list)
        output: results (dictionary)

        This function performs a grid search over the specified hyperparameters and returns the best combination
        """

        # Generate all possible combinations of hyperparameters
        param_combinations = list(ParameterGrid(param_grid))

        results = []

        # Iterate over each combination of hyperparameters
        for params in param_combinations:
            for num_epochs in num_epochs_list:


                # Create a new instance of your model with the specified hyperparameters
                ann_model = Ann(self.df, self.target_variable)

                # Train the model
                ann_model.train_model(epochs=num_epochs, batch_size=params['batch_size'])  # Pass batch_size separately

                # Evaluate the model
                evaluation_result = ann_model.evaluate()

                results.append({'params': params, 'num_epochs': num_epochs, 'evaluation_result': evaluation_result})
        
        # we return the results with the lowest evaluation result
        
        return min(results, key=lambda x: x['evaluation_result'])

    def compute_feature_importance(self):
        """
        input: None
        output: feature_importance (list)

        This function computes the feature importance of each feature
        """

        self.eval()
        self.X_test.requires_grad = True
        dataset = TensorDataset(self.X_test.view(-1, self.window_size * self.X_test.size(2)), self.y_test)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Compute the loss without any feature
        loss_without_feature = 0
        for X_batch, y_batch in dataloader:
            self.optimizer.zero_grad()
            predictions = self(X_batch)
            loss_without_feature += self.criterion(predictions, y_batch)
        loss_without_feature = loss_without_feature / len(dataloader)

        # Compute the loss with each feature
        feature_importance = []
        for feature_idx, feature_name in enumerate(self.df.drop(self.target_variable, axis=1).columns):
            loss_with_feature = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                predictions = self(X_batch)
                loss_with_feature += self.criterion(predictions, y_batch)
            loss_with_feature = loss_with_feature / len(dataloader)
            importance_value = (loss_without_feature - loss_with_feature) / loss_without_feature
            feature_importance.append((feature_name, importance_value.item()))

        return feature_importance
        