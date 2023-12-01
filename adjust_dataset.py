# This file will adjust a dataset following the data analysis done previously.
# It will receive a dataframe and a location and will return a dataframe with the adjusted data.
import pandas as pd

def adjust_dataset(df, location):
    
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
    # columns that are constant
    constant_cols = ['country', 'timezone', 'latitude', 'longitude', 'last_updated_epoch']
    df.drop(columns=highly_correlated_cols, inplace=True)
    df.drop(columns=constant_cols, inplace=True)
    # we will also take the location out of the dataset since we already used it and it is constant
    df.drop(columns=['location_name'], inplace=True)



    # 4. Apply one hot encoding for categorical columns (but not to dates)
    categorical_variables = df.select_dtypes(include=['object']).columns
    categorical_variables = categorical_variables.drop(['sunrise', 'sunset', 'moonrise', 'moonset'])
    # we will also drop condition_text since it is a target variable
    categorical_variables = categorical_variables.drop(['condition_text'])
    df = pd.get_dummies(df, columns=categorical_variables)
    

    # 5. There are instantes with moonset and moonrise where there are missing values, because there is none. We 
    # will fill those values with with the last value of the column
    df['moonset'].replace('No moonset', method='ffill', inplace=True)
    df['moonrise'].replace('No moonrise', method='ffill', inplace=True)

    # 6. Change format of objetc columns to datetime
    df['sunrise'] = pd.to_datetime(df['sunrise'])
    df['sunset'] = pd.to_datetime(df['sunset'])
    df['moonrise'] = pd.to_datetime(df['moonrise'])
    df['moonset'] = pd.to_datetime(df['moonset'])

    #  the end result is a dataframe with 47 columns

    
    return df


# df = pd.read_csv('./docs/data/GlobalWeatherRepository.csv')
# df = adjust_dataset(df, 'Madrid')

# # lets see the unique values of Moonries 
# print(df['moonrise'].unique())