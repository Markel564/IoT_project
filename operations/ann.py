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

        for city in self.df['location_name'].unique():
            city_data = self.df[self.df['location_name'] == city]

            for i in range(len(city_data) - self.window_size - self.output_size):
                seq = city_data.iloc[i: i + self.window_size].drop([self.target_variable, 'location_name'], axis=1)
                sequences.append(seq.values)
        print ("Cantidad de windows: ", len(sequences))        
        self.X_train = torch.tensor(sequences[:int(len(sequences) * train_size)], dtype=torch.float32)
        self.X_test = torch.tensor(sequences[int(len(sequences) * train_size):], dtype=torch.float32)

    def y_conversion(self, train_size):
        labels = []
        for city in self.df['location_name'].unique():
            city_data = self.df[self.df['location_name'] == city]

            for i in range(self.window_size, len(city_data) - self.output_size):
                label = city_data.iloc[i: i + self.output_size][self.target_variable]
                labels.append(label.values)
        print ("Cantidad de labels: ", len(labels))
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
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32, self.output_size)
        )

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
    
    def train_model(self, epochs=50, batch_size=32):
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

    def compute_feature_importance(self):

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