from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import os


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