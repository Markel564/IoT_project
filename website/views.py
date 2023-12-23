from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify
from .python_code.city import get_city_and_country
from .python_code.time import get_date, get_hour
from .python_code.current_data import get_data, create_data
from .python_code.adjust_dataset import adjust_dataset
import pandas as pd
from .python_code.get_models_and_paths import get_models, get_paths

views = Blueprint('views', __name__)


target_variables = ['temperature_celsius', 'humidity', 'precip_mm', 'cloud', 'wind_kph']


# models = get_models(target_variables, paths)


@views.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':        
        
        selected_city = request.form.get('selectedCity')

        selectedAlgorithm = request.form.get('selectedAlgorithm')
        
        if not selected_city:

            flash('Please select a city', category = 'error')
            return redirect(url_for('views.home'))
        
        return redirect(url_for('views.page', city = selected_city, algorithm = selectedAlgorithm))

        

    else:
        return render_template('home.html')


@views.route('/page', methods = ['GET', 'POST'])
def page():
    if request.method == 'POST':    #go back to home
        
        button = request.form.get('button')
        

        return redirect(url_for('views.home'))

    else:
        
        # We get city and algorithm from the request
        city = request.args.get('city')
        algorithm = request.args.get('algorithm')
        # we get the date and time
        date = get_date()
        time = get_hour()

        
        # we get the data for the selected city
        target_variables = ['temperature_celsius', 'humidity', 'precip_mm', 'cloud', 'wind_kph']

        # each dataset differes for each target variable
        datasets = []

       


        # we dowlnoad the data
        create_data()
        for target_variable in target_variables:
            # and we get the last 7 days's dataset for each target variable
            datasets.append(get_data(city, target_variable))

        

        # we get the paths
        paths = get_paths(target_variables, algorithm)

        
        # and the models
        print ("Creating models with algorithm: ", algorithm)
        models = get_models(target_variables, algorithm, paths)
        print ("Models created")
        


        # lets print the predictions
        predictions = []
        for i in range(5):
            data_right_now = get_data(city, target_variables[i])
            predictions.append(models[i].predict(data_right_now))
            print (predictions[i])


        
        # we get the city and the country of the city
        city = get_city_and_country(city)


        # if there is no algorithm (because it is predefined as ANN):
        if not algorithm:
            return render_template('page.html', city = city, algorithm = "ANN", date=date, time=time)
        else:
            
            return render_template('page.html', city = city, algorithm = algorithm, date=date, time=time)


@views.route('/signal', methods = ['GET', 'POST'])
def signal():
    if request.method == 'POST':
        
        # a json was sent
        data = request.get_json() 
        info_lstm = """
        ROOT MEAN SQUARE ERRORS\n
        temperature: 0.000000\n
        humidity: 0.000000\n
        precipitation: 0.000000\n
        cloudiness: 0.000000\n
        wind speed: 0.000000\n
        """
        print (data["algorithm"])
        flash(info_lstm, category = 'success')
        return jsonify({"message": "Signal processed successfully"})

    else:
        return redirect(url_for('views.home'))




