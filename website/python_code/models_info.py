"""
This file returns a string with the information (rmse) of each model
for each target variable (temperature, humidity, pressure, wind speed, wind direction)
"""


def rmse_info(algorithm):

    if algorithm == "ANN":

        return """
        temperature     --> 3.24
        wind speed      --> 7.8
        humidity        --> 16.4
        precipitation   --> 0.74
        cloudiness      --> 31.3

        """ 
    else:
        return """
        temperature     --> 3.78
        wind speed      --> 8.0
        humidity        --> 18.5
        precipitation   --> 0.73
        cloudiness      --> 31.5
        
        """