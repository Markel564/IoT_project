from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
# Ruta al archivo JSON de tu token API de Kaggle dentro del proyecto
kaggle_json_path = "./docs/kaggle.json"

# Directorio predeterminado donde Kaggle busca el archivo kaggle.json
default_kaggle_dir = "/home/markel/.kaggle"

# Copia el archivo kaggle.json al directorio predeterminado
shutil.copy(kaggle_json_path, default_kaggle_dir)

# Configuración de la API de Kaggle
api = KaggleApi()
api.authenticate()

# Nombre del conjunto de datos en Kaggle
dataset_name = "nelgiriyewithana/global-weather-repository"

# Directorio de destino para el conjunto de datos descargado
output_dir = "./docs/data/"

# Descargar el conjunto de datos
api.dataset_download_files(dataset_name, path=output_dir, unzip=True)

print("Downloaded the dataset to " + output_dir + " successfully!")