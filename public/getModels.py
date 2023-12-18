import firebase_admin
from firebase_admin import credentials, storage

def download_model(model_name, local_path):
    # Initialize Firebase if it hasn't been initialized yet
    if not firebase_admin._apps:
        cred = credentials.Certificate('path/to/serviceAccountKey.json')
        firebase_admin.initialize_app(cred, {
            'weather-appli-7d3e5.appspot.com': 'gs://weather-appli-7d3e5.appspot.com'
        })

    bucket = storage.bucket()
    blob = bucket.blob(model_name)
    blob.download_to_filename(local_path)
    print(f'Model {model_name} downloaded to {local_path}')

# Example usage
download_model('path/to/your/model', 'local/path/to/save/model')
