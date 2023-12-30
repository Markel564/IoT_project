This project consists of the implementation a website that uses machine learning algorithms for weather
forecast. 

The algorithms implemented are:

- Long Short Term Memory (LSTM)
- Artificial Neural Network (ANN)

Such algorithms have been trained with data provided by the reposiroty found in the following link:

https://www.kaggle.com/code/nelgiriyewithana/an-introduction-to-global-weather-repository


Finally, a website has been created, which allows the user to select the city and algorithm to be used for the forecast. The
goal is that this website is used for educational purposes, so that users can compare the results of the different algorithms.

The website has been created using the Flask framwork. The link for the website is the following:

https://jompish.pythonanywhere.com/

In a future, we would like to implement more algorithms and allow the user to select the parameters of the algorithms, as well as
selecting a wider range of cities.

It is also important to highlight that the purpose of the website is to show how ANN and LSTM work, and not to provide a reliable
weather forecast. Hence, if a result is predicted as negative and is impossible (for example, precipitation), it has been left as
negative and not corrected to 0, as this would be correcting the algorithm's prediction.

If you want to run the website locally, first run the requirements.txt file to install the required libraries

pip install -r requirements.txt

Then, run the following command:

python main.py

For more information about the project, a project report has been written and is included in the repository.


Johannes Kvärnstrom
Markel Benedicto

