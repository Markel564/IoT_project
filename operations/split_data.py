import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_data(df, target_variable, train_size):

    train_df = df[:train_size]
    test_df = df[train_size:]

    target_variables = ['temperature_celsius', 'wind_kph', 'humidity', 'pressure_mb', 'precip_mm', 'cloud']

    if target_variable not in target_variables:
        raise ValueError(f"Target variable must be one of {target_variables}")


    X_train = train_df.drop(columns=target_variable) # we drop the target variable from the training set
    y_train = train_df[target_variable] # we keep only the target variable in the training set

    X_test = test_df.drop(columns=target_variable) 
    y_test = test_df[target_variable] 

    # Now, we will scale the data

    scaler = StandardScaler()

    # we will scale only numerical values

    numerical_columns = X_train.select_dtypes(include=['int', 'float']).columns

    scaler.fit(X_train[numerical_columns])
    scaler.fit(X_test[numerical_columns])

    X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])

    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    return X_train, y_train, X_test, y_test