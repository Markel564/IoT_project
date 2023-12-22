from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify
from .python_code.city import get_city_and_country
from .python_code.time import get_date, get_hour
views = Blueprint('views', __name__)


@views.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':        
        
        selected_city = request.form.get('selectedCity')

        selectedAlgorithm = request.form.get('selectedAlgorithm')
        
        if not selected_city:

            flash('Please select a city', category = 'error')
            return redirect(url_for('views.index'))
        
        return redirect(url_for('views.page', city = selected_city, algorithm = selectedAlgorithm))

        

    else:
        return render_template('index.html')


@views.route('/page', methods = ['GET', 'POST'])
def page():
    if request.method == 'POST':
        
        button = request.form.get('button')
        print(button)

        return redirect(url_for('views.index'))

    else:
        
        city = request.args.get('city')
        algorithm = request.args.get('algorithm')

        # we get the city and the country of the city
        city = get_city_and_country(city)

        date = get_date()
        time = get_hour()

        print ("date and time are", date, time)
        print ("City is", city)
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
        return redirect(url_for('views.index'))