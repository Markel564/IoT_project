from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import os
import pandas as pd

# route to kaggle.json
kaggle_json_path = "./docs/kaggle.json"

# default kaggle.json path
user_home_dir = os.path.expanduser("~")
default_kaggle_dir = os.path.join(user_home_dir, ".kaggle")

# copy kaggle.json to default kaggle.json path
shutil.copy(kaggle_json_path, default_kaggle_dir)

# configure kaggle api
api = KaggleApi()
api.authenticate()

# name of the dataset
dataset_name = "nelgiriyewithana/global-weather-repository"

# output directory
output_dir = "./docs/data/"

# download the dataset
api.dataset_download_files(dataset_name, path=output_dir, unzip=True)

print("Downloaded the dataset to " + output_dir + " successfully!")



# lets adjust the dataset for each target variable


target_variables = ["temperature_celsius", "humidity", "wind_kph", "precip_mm", "cloud"]

from operations.adjust_dataset import adjust_dataset

for target_variable in target_variables:
    
    df = pd.read_csv("./docs/data/GlobalWeatherRepository.csv")
    df = adjust_dataset(df, target_variable)

    # we save the dataset in docs/data/{target_variable}
    # with the name GlobalWeatherRepository_target_variable.csv

    dataset = df.to_csv("./docs/data/GlobalWeatherRepository_{}.csv".format(target_variable), index=False)

    target_folder = "./docs/data/{}".format(target_variable)

    file_path = os.path.join(target_folder, "GlobalWeatherRepository_{}.csv".format(target_variable))
    
        



