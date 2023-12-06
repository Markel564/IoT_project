# This file will adjust a dataset following the data analysis done previously.
# It will receive a dataframe and a location and will return a dataframe with the adjusted data.

# To eliminate warnings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.preprocessing import StandardScaler


def adjust_dataset(df, location, target_variable):
    
    """
    df: dataframe to be adjusted
    location: location that the user has selected to predict the weather

    return: adjusted dataframe
    """

    # 1. Keep only those rows that belong to the location selected by the user
    df = df[df['location_name'] == location].copy()

    # 2. Order by last_updated and eliminate it
    df.sort_values(by=['last_updated'], inplace=True)
    df.drop(columns=['last_updated'], inplace=True)

    # 3. Columns that are not useful for the prediction
    # columns that are highly correlated with other columns
    highly_correlated_cols = ['temperature_fahrenheit', 'wind_mph', 'pressure_in', 'precip_in', 'gust_mph', 'feels_like_fahrenheit', 'visibility_miles', 'air_quality_us-epa-index']
    # columns that are constant or useless.
    constant_cols = ['country', 'timezone', 'latitude', 'longitude', 'last_updated_epoch']
    df.drop(columns=highly_correlated_cols, inplace=True)
    df.drop(columns=constant_cols, inplace=True)
    # we will also take the location out of the dataset since we already used it and it is constant
    df.drop(columns=['location_name'], inplace=True)



    # 4. Apply one hot encoding for categorical columns (but not to dates)
    categorical_variables = df.select_dtypes(include=['object']).columns
    categorical_variables = categorical_variables.drop(['sunrise', 'sunset', 'moonrise', 'moonset'])
    df = pd.get_dummies(df, columns=categorical_variables)
    

    # 5. There are instantes with moonset and moonrise where there are missing values, because there is none. We 
    # will fill those values with with the last value of the column
    df['moonset'].replace('No moonset', method='ffill', inplace=True)
    df['moonrise'].replace('No moonrise', method='ffill', inplace=True)

    # 6. Change format of objetc columns to timestamp
    df['sunrise'] = pd.to_datetime(df['sunrise']).apply(lambda x: x.timestamp())
    df['sunset'] = pd.to_datetime(df['sunset']).apply(lambda x: x.timestamp())
    df['moonrise'] = pd.to_datetime(df['moonrise']).apply(lambda x: x.timestamp())
    df['moonset'] = pd.to_datetime(df['moonset']).apply(lambda x: x.timestamp())

    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns
    numerical_columns.drop(target_variable)

    scaler.fit(df[numerical_columns])
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    # 7. Change all boolean columns to 0 and 1 (not sure if this is necessary)
    
    boolean_columns = df.select_dtypes(include=['bool']).columns
    
        
    for column in boolean_columns:
        df[column] = df[column].astype(int)


    

    
    return df

