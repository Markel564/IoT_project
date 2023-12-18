from flask import Blueprint, render_template, request, flash, redirect, url_for, session

views = Blueprint('views', __name__)


@views.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':        
        
        selected_city = request.form.get('selectedCity')

        selectedAlgorithm = request.form.get('selectedAlgorithm')
        
        if not selected_city:

            flash('Please select a city', category = 'error')
            return redirect(url_for('views.home'))
        
        print ("selected city is", selected_city)
        print ("selected algo is", selectedAlgorithm)
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
        
        return render_template('page.html', city = request.args.get('city'), algorithm = request.args.get('algorithm'))


