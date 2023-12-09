from flask import Blueprint, render_template, request, flash, redirect, url_for, session

views = Blueprint('views', __name__)


@views.route('/', methods = ['GET', 'POST'])
def home():

    if request.method == 'POST':        
        pass

    else:
        return render_template('home.html')


