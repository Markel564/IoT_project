from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import pandas as pd
from ann_model import Ann
from adjust_dataset import adjust_dataset

# Example of loading the model
# You will need to adjust this based on how your model expects to receive the dataframe and target variable
#df = pd.read_csv('././docs/data/GlobalWeatherRepository.csv')
#target_variable = 'temperature_celsius' # Define your target variable
#df_adjusted = adjust_dataset(df, target_variable)

#model = Ann(df, target_variable)
#model.load_state_dict(torch.load('models/temperature_celsius_ann_model.pth'))
#model.eval()

## Filter the DataFrame for rows where location_name is 'Madrid'
#madrid_df = df_adjusted[df_adjusted['location_name'] == 'Madrid']

# Select the first 7 rows of the filtered DataFrame
#sample_df = madrid_df.head(7)
# Ensure all data is numeric and handle non-numeric data if any
#sample_df = sample_df.apply(pd.to_numeric, errors='coerce')
# Now pass the DataFrame to the predict method
#prediction = model.predict(sample_df)

#df = pd.read_csv('../docs/data/GlobalWeatherRepository.csv')


df = pd.read_csv('././docs/data/GlobalWeatherRepository.csv')
target_variable = 'temperature_celsius'

df = adjust_dataset(df, target_variable=target_variable)
print (df.shape)

model = Ann(df, target_variable=target_variable)
path = "././models/temperature_celsius_ann_model.pth"
#model = torch.load(path)

model.load_state_dict(torch.load(path))
print(type(model))
#model.train_model()

df = pd.read_csv('././docs/data/GlobalWeatherRepository.csv')
df = df[df["location_name"] == "Madrid"]

df.shape

df = df.sort_values(by="last_updated", ascending=False).head(7)

df.shape

final_df = adjust_dataset(df, "temperature_celsius")
df.shape

# lets drop location_name and last_updated
final_df = final_df.drop(columns=["location_name"])

final_df.shape
pred = model.predict(final_df)

print (pred)

app = Flask(__name__)


@app.route('/')
def home():
    # Option 1: Display a welcome message
    return "Welcome to the Flask App!"

    # Option 2: Redirect to the form
    # from flask import redirect, url_for
    # return redirect(url_for('show_form'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # For testing, using 'Madrid'. Replace with 'city' if using dynamic input
        prediction = model.predict(final_df)
    except Exception as e:
        prediction = f"ErrorJag JApp dig: {str(e)}"  # Display actual error message

    return render_template('prediction.html', prediction=prediction)

@app.route('/form', methods=['GET'])
def show_form():
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)