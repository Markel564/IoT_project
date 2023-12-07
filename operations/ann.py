import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid

torch.manual_seed(42)

class Ann(nn.Module):
    def __init__(self, df, target_variable):
        super(Ann, self).__init__()

        self.df = df
        self.window_size = 7
        self.output_size = 3
        self.target_variable = target_variable
        
        self.x_conversion(0.7)
        self.y_conversion(0.7)
        self.shuffle_data()
        self.build_model()

    def x_conversion(self, train_size):
        sequences = []
        for i in range(len(self.df) - self.window_size - self.output_size):
            seq = self.df.drop(self.target_variable, axis=1).iloc[i : i + self.window_size]
            sequences.append(seq.values)

        self.X_train = torch.tensor(sequences[:int(len(sequences) * train_size)], dtype=torch.float32)
        self.X_test = torch.tensor(sequences[int(len(sequences) * train_size):], dtype=torch.float32)

    def y_conversion(self, train_size):
        labels = []
        for i in range(self.window_size, len(self.df) - self.output_size):
            label = self.df[self.target_variable].iloc[i : i + self.output_size]
            labels.append(label.values)

        self.y_train = torch.tensor(labels[:int(len(labels) * train_size)], dtype=torch.float32)
        self.y_test = torch.tensor(labels[int(len(labels) * train_size):], dtype=torch.float32)

        
    def shuffle_data(self):
        indices = torch.randperm(len(self.X_train))
        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]

    def build_model(self):
        # {'params': {'batch_size': 32, 'dropout': 0.2, 'lr': 0.01, 'lr_schedule': 'exponential_decay', 
        # 'neurons_per_layer': 32, 'num_layers': 2, 'optimizer': 'RMSprop'}, 'num_epochs': 100,
        #  lets build a model with these parameters
        self.model = nn.Sequential(
            nn.Linear(self.window_size * self.X_train.size(2), 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32, self.output_size)
        )

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        # self.model = nn.Sequential(
        #     nn.Linear(self.window_size * self.X_train.size(2), 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(16, self.output_size)
        # )

        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)
        # self.criterion = nn.MSELoss()
    
    def train_model(self, epochs=100, batch_size=32):
        dataset = TensorDataset(self.X_train.view(-1, self.window_size * self.X_train.size(2)), self.y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                predictions = self(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        with torch.no_grad():
            predictions = self(self.X_test.view(-1, self.window_size * self.X_test.size(2)))
            test_loss = self.criterion(predictions, self.y_test)
            rmse = torch.sqrt(test_loss)

        return rmse.item()

    
    
    def predict(self, df):
        X = df.drop(self.target_variable, axis=1).iloc[-self.window_size:].values
        X = torch.tensor(X, dtype=torch.float32).reshape(1, -1, self.window_size * self.X_train.size(2))
        with torch.no_grad():
            predictions = self(X)

        return predictions[0].numpy()


    def forward(self, x):
        return self.model(x)


    def hyperparameter_tuning(self, param_grid, num_epochs_list=[10, 20]):

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