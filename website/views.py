"""
This file contains the views of the website.

There are two views:
    - home: corresponds to the home page
    - page: corresponds to the prediction page

The home page is the one that is shown when we run the command "python main.py" in the terminal.
It allows the user to select a city and an algorithm.

The prediction page is the one that is shown when we click the button "GO" in the home page.
It shows the prediction of the weather for the next 3 days for the selected city and algorithm.
"""
from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify
from .python_code.city import get_city_and_country
from .python_code.time import get_date, get_hour, get_dates
from .python_code.current_data import get_data, create_data
import pandas as pd
from .python_code.get_models_and_paths import get_models, get_paths
from .python_code.condition import condition
from .python_code.pack_info import pack_info
from .python_code.models_info import rmse_info

views = Blueprint('views', __name__)

# the target variables to be used
target_variables = ['temperature_celsius', 'humidity', 'precip_mm', 'cloud', 'wind_kph']



# This corresponds to the home page
@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':    # if there is a post request --> go to the prediction page
        
        # we get the selected city and algorithm
        selected_city = request.form.get('selectedCity')
        selectedAlgorithm = request.form.get('selectedAlgorithm')
        
        # if the user does not select a city, we send an error message
        if not selected_city:
            flash('Please select a city', category='error')
            return redirect(url_for('views.home'))

        # redirect to the prediction page
        return redirect(url_for('views.page', city=selected_city, algorithm=selectedAlgorithm))

    else:
        # if it is a get request, we just render the home page

        # obtain the rmse info of the models
        info_ANN = rmse_info("ANN")
        info_LSTM = rmse_info("LSTM")
        
        # render the home page
        return render_template('home.html', info_ANN = info_ANN, info_LSTM = info_LSTM)


# This corresponds to the prediction page
@views.route('/page', methods = ['GET', 'POST'])
def page():
    #go back to home page if there is a post request
    if request.method == 'POST':    
        
        # the only post request available is if the user clicks the back button

        return redirect(url_for('views.home'))

    else:  # if it is a get request --> render the prediction page
        
        # We get city and algorithm from the request
        city = request.args.get('city')
        algorithm = request.args.get('algorithm')

        # there is a chance the user does not select an algorithm (by default is ANN)
        if not algorithm:
            algorithm = "ANN"
        # we get the date and time
        result = get_hour(city)
        print (result)
        time = result[1]

        date = get_date(result[0])
        
        
            
        # we get the data for the selected city
        target_variables = ['temperature_celsius', 'humidity', 'precip_mm', 'cloud', 'wind_kph']

        # each dataset differes for each target variable (for speed purposes)
        datasets = []


        # download the datasets
        create_data()
        for target_variable in target_variables:
            # and we get the last 7 days's dataset for each target variable
            datasets.append(get_data(city, target_variable))
    
        #  get the paths of each model
        paths = get_paths(target_variables, algorithm)

        
        # and the models itself
        print ("Creating models with algorithm: ", algorithm)
        models = get_models(target_variables, algorithm, paths)
        print ("Models created")
        

        # predict the weather for each target variable
        predictions = []
        for i in range(5):
            data_right_now = get_data(city, target_variables[i])
            predictions.append(models[i].predict(data_right_now))
        
        # since the predict method for each algorithm returns a different shape, we need to select
        # the correct values depending on the algorithm
        if algorithm == "ANN":
            condition_1 = condition(predictions[0][0][0], predictions[1][0][0], predictions[2][0][0], predictions[3][0][0], predictions[4][0][0])
            condition_2 = condition(predictions[0][0][1], predictions[1][0][1], predictions[2][0][1], predictions[3][0][1], predictions[4][0][1])
            condition_3 = condition(predictions[0][0][2], predictions[1][0][2], predictions[2][0][2], predictions[3][0][2], predictions[4][0][2])

            info = pack_info(condition_1, condition_2, condition_3, predictions[0][0][0], predictions[0][0][1], predictions[0][0][2],
            predictions[1][0][0], predictions[2][0][0], predictions[4][0][0])
            
        else: # if it is LSTM
            condition_1 = condition(predictions[0][0], predictions[1][0], predictions[2][0], predictions[3][0], predictions[4][0])
            condition_2 = condition(predictions[0][1], predictions[1][1], predictions[2][1], predictions[3][1], predictions[4][1])
            condition_3 = condition(predictions[0][2], predictions[1][2], predictions[2][2], predictions[3][2], predictions[4][2])

            info = pack_info(condition_1, condition_2, condition_3, predictions[0][0], predictions[0][1], predictions[0][2],
            predictions[1][0], predictions[2][0], predictions[4][0])


    
        # we format the city and the country of the city
        city = get_city_and_country(city)


        # we also will send the dates (monday, tuesday, ...) to the page
        dates = get_dates(result[0])
        
        
        # and render the page with the information obtained (city, algorithm, date, time, info, dates) 
        return render_template('page.html', city = city, algorithm = algorithm, date=date, time=time, info = info, dates = dates)
        
            
            




