# This file will adjust a dataset following the data analysis done previously.
# It will receive a dataframe and a location and will return a dataframe with the adjusted data.

# To eliminate warnings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

def adjust_dataset(df,target_variable):
    
    """
    df: dataframe to be adjusted
    location: location that the user has selected to predict the weather

    return: adjusted dataframe
    """
    
    # 1. Order by location_name
    df.sort_values(by=['location_name', 'last_updated'], inplace=True)
    # 2. Order by last_updated and eliminate it since we dont need them
    df.drop(columns=['last_updated'], inplace=True)

    # 3. Columns that are not useful for the prediction
    # columns that are highly correlated with other columns
    highly_correlated_cols = ['temperature_fahrenheit', 'wind_mph', 'pressure_in', 'precip_in', 'gust_mph', 'feels_like_fahrenheit', 'visibility_miles', 'air_quality_us-epa-index']
    # columns that are constant or useless.
    constant_cols = ['country', 'timezone', 'latitude', 'longitude', 'last_updated_epoch']
    df.drop(columns=highly_correlated_cols, inplace=True)
    df.drop(columns=constant_cols, inplace=True)


    df.drop(columns=['condition_text'], inplace=True)


    


    # 4. Apply one hot encoding for categorical columns (but not to dates)
    categorical_variables = df.select_dtypes(include=['object']).columns
    categorical_variables = categorical_variables.drop(['sunrise', 'sunset', 'moonrise', 'moonset', 'location_name'])



    # CHANGES
    # df = pd.get_dummies(df, columns=categorical_variables)
    df.drop(columns=categorical_variables, inplace=True)


    # 5. There are instantes with moonset and moonrise where there are missing values, because there is none. We 
    # will delete those rows
    df['moonset'] = df['moonset'].replace('No moonset', pd.NaT).fillna(method='bfill')
    df['moonrise'] = df['moonrise'].replace('No moonrise', pd.NaT).fillna(method='bfill')
    


    # # 6 Change format of object columns to timestamp
    df['sunrise'] = pd.to_datetime(df['sunrise']).apply(lambda x: x.timestamp() if not pd.isna(x) else x)
    df['sunset'] = pd.to_datetime(df['sunset']).apply(lambda x: x.timestamp() if not pd.isna(x) else x)
    df['moonrise'] = pd.to_datetime(df['moonrise']).apply(lambda x: x.timestamp() if not pd.isna(x) else x)
    df['moonset'] = pd.to_datetime(df['moonset']).apply(lambda x: x.timestamp() if not pd.isna(x) else x)


    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns
    numerical_columns = numerical_columns.drop(target_variable)

    # numerical_columns = numerical_columns.drop(['sunrise', 'sunset', 'moonrise', 'moonset'])
    scaler.fit(df[numerical_columns])
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    

    # 7. Change all boolean columns to 0 and 1 (not sure if this is necessary)
    
    boolean_columns = df.select_dtypes(include=['bool']).columns
        
    for column in boolean_columns:
        df[column] = df[column].astype(int)

    # 8. Eliminate outliers
    z_scores = zscore(df[numerical_columns])
    abs_z_scores = abs(z_scores)
    outliers = (abs_z_scores > 3).all(axis=1)
    df_no_outliers = df[~outliers]

  
    return df_no_outliers

