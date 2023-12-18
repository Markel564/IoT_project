from flask import Blueprint, render_template, request, flash, redirect, url_for, session
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
            return redirect(url_for('views.home'))
        
        return redirect(url_for('views.page', city = selected_city, algorithm = selectedAlgorithm))

        

    else:
        return render_template('home.html')


@views.route('/page', methods = ['GET', 'POST'])
def page():
    if request.method == 'POST':
        
        button = request.form.get('button')
        print(button)

        return redirect(url_for('views.home'))

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


