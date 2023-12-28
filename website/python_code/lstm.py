"""
-- This file contains the implementation of the LSTM model --

For more description of the model, please check the files at operations/lstm
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)  # Add dropout layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)  # Apply dropout
        output = self.fc(lstm_out[:, -1, :])
        return output

        
class LSTMPredictorWrapper:
    def __init__(self, df, target_variable, hidden_size=32, num_layers=2, dropout=0.2):
        """
        Constructor of the LSTM class
        """
        self.df = df    # receives a full dataframe standarized and adjusted
        self.window_size = 7 # 7 days predict 3 days
        self.output_size = 3
        self.target_variable = target_variable
        self.hidden_size = hidden_size # Number of hidden units
        self.num_layers = num_layers # Number of LSTM layers
        self.dropout = dropout # Dropout rate

        self.x_conversion(0.7) # 70% of the data is used for training
        self.y_conversion(0.7)
        self.shuffle_data() # shuffle the data (windows, so order is maintained)
        self.build_model()

    def load_model(self, filepath):
        """
        Loads the internal LSTMPredictor model from the specified filepath.
        """
        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def save_model(self, filepath):
        """
        Saves the internal LSTMPredictor model to the specified filepath.
        """
        torch.save(self.model.state_dict(), filepath)

    def x_conversion(self, train_size):
        """
        input: train_size (float)
        output: None

        This function converts the input dataframe into a tensor of shape (num_samples, window_size, num_features)
        """
        sequences = []

        for city in self.df['location_name'].unique():
            city_data = self.df[self.df['location_name'] == city]

            for i in range(len(city_data) - self.window_size - self.output_size + 1):
                seq = city_data.iloc[i: i + self.window_size].drop([self.target_variable, 'location_name'], axis=1)

                sequences.append(seq.values)

        self.X_train = torch.tensor(sequences[:int(len(sequences) * train_size)], dtype=torch.float32)
        self.X_test = torch.tensor(sequences[int(len(sequences) * train_size):], dtype=torch.float32)


    def y_conversion(self, train_size):
        """
        input: train_size (float)
        output: None

        This function converts the input dataframe into a tensor of shape (num_samples, window_size, num_features)
        """
        labels = []
        for city in self.df['location_name'].unique():
            city_data = self.df[self.df['location_name'] == city]

            for i in range(self.window_size, len(city_data) - self.output_size + 1):
                label = city_data.iloc[i: i + self.output_size][self.target_variable]
                labels.append(label.values)
        self.y_train = torch.tensor(labels[:int(len(labels) * train_size)], dtype=torch.float32)
        self.y_test = torch.tensor(labels[int(len(labels) * train_size):], dtype=torch.float32)

    def shuffle_data(self):
        """
        input: None
        output: None

        This function shuffles the data for a better prediction
        """
        indices = torch.randperm(len(self.X_train)) # Random permutation of indices
        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]

    def build_model(self):
        """
        input: None
        output: None

        This function builds the LSTM model
        """
        self.model = LSTMPredictor(
            input_size=self.X_train.size(2),
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def train_model(self, epochs=10, batch_size=32, max_grad_norm=2.0):
        """
        input: epochs (int), batch_size (int), max_grad_norm (float)
        output: None

        This function trains the LSTM model
        """
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_losses = []

        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()

                train_losses.append(loss.item())


    def evaluate(self):
        """
        input: None
        output: rmse (float)

        This function evaluates the LSTM model
        """
        with torch.no_grad():
            predictions = self.forward(self.X_test)  # Fix this line
            test_loss = self.criterion(predictions, self.y_test)
            rmse = torch.sqrt(test_loss)
        return rmse.item()

    def predict(self, df):
        """
        input: df (pd.DataFrame) the dataframe to predict containing 7 days
        output: predictions (array) of 3 days, in the format array([day1, day2, day3])

        This function predicts the next 3 days of the input dataframe
        """
        X = df.drop(self.target_variable, axis=1).iloc[-self.window_size:].values
        X = torch.tensor(X, dtype=torch.float32).reshape(1, -1, self.X_train.size(2))
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions[0].numpy()

    def forward(self, x):
        """
        input: x (tensor)
        output: predictions (tensor)

        This function makes a forward pass of the model
        """
        return self.model(x)

    def hyperparameter_tuning(self, param_grid, num_epochs_list):
        """
        input: param_grid (dict), num_epochs_list (list)
        output: best_params (dict)

        This function performs a grid search to find the best hyperparameters
        """
        param_combinations = list(ParameterGrid(param_grid))
        results = []

        for params in param_combinations:
            for num_epochs in num_epochs_list:
                lstm_model = LSTMPredictorWrapper(
                    df=self.df,
                    target_variable=self.target_variable,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers']
                )

                lstm_model.train_model(epochs=num_epochs, batch_size=params['batch_size'])
                evaluation_result = lstm_model.evaluate()

                results.append({'params': params, 'num_epochs': num_epochs, 'evaluation_result': evaluation_result})

        return min(results, key=lambda x: x['evaluation_result'])

    def compute_feature_importance(self):
        """
        input: None
        output: None

        This function computes the feature importance of the model. In the case of LSTMs, this is not implemented,
        as the feature importance is not easily computed. 
        """
        raise NotImplementedError("Feature importance computation is not implemented for LSTMs.")


