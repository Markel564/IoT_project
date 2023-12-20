from flask import Flask, request, jsonify
import torch
from ann_model import Ann


app = Flask(__name__)

# Load your model
model = your_model.YourModelClass()
model.load_state_dict(torch.load('models/cloud_ann_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    city = data['city']
    # Perform prediction based on the city
    # You'll need to implement how the city data is converted to a format your model can use
    prediction = model.predict(city_data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)