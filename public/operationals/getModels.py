import firebase_admin
from firebase_admin import credentials, storage

import os
print("Current working directory:", os.getcwd())

def download_model(model_name, local_path):
    # Initialize Firebase if it hasn't been initialized yet
    if not firebase_admin._apps:
        cred = credentials.Certificate('public/operationals/weather-appli-7d3e5-firebase-adminsdk-mov3t-336da32f64.json')
        firebase_admin.initialize_app(cred, {
             'storageBucket': 'weather-appli-7d3e5.appspot.com'
        })

    bucket = storage.bucket()
    blob = bucket.blob(model_name)
    blob.download_to_filename(local_path)
    print(f'Model {model_name} downloaded to {local_path}')

# Example usage
# Download cloud models
download_model('cloud_ann_model.pth', './models/cloud_ann_model.pth')
download_model('cloud_lstm_model.pth', './models/cloud_lstm_model.pth')

# Download humidity models
download_model('humidity_ann_model.pth', './models/humidity_ann_model.pth')
download_model('humidity_lstm_model.pth', './models/humidity_lstm_model.pth')

# Download precip_mm models
download_model('precip_mm_ann_model.pth', './models/precip_mm_ann_model.pth')
download_model('precip_mm_lstm_model.pth', './models/precip_mm_lstm_model.pth')

# Download temperature_celsius models
download_model('temperature_celsius_ann_model.pth', './models/temperature_celsius_ann_model.pth')
download_model('temperature_celsius_lstm_model.pth', './models/temperature_celsius_lstm_model.pth')

# Download wind_kph models
download_model('wind_kph_ann_model.pth', './models/wind_kph_ann_model.pth')
download_model('wind_kph_lstm_model.pth', './models/wind_kph_lstm_model.pth')


