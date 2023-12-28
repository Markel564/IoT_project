"""
This file contains a function that returns
the condition of the weather based on the input.

It is arbirtrary and it is not based on any scientific
evidence, it is just a way to show the user the weather condition

"""



def condition(temp, humidity, precip, cloud, wind):
    
    """
    options are:
    - sunny
    - cloudy            
    - rainy             
    - snowy             
    - windy             
    - cloudy-sunny      
    """

    print ("temp: ", temp, "precip: ", precip, "cloud: ", cloud, "humidity: ", humidity, "wind: ", wind)
    if wind >= 15:
        return "WINDY"  
    if precip >= 1 and temp >= 0:
        return "RAINY"
    if temp <= 0 and precip >= 1:
        return "SNOWY"
    if cloud >= 50: # precip is low so it does not rain
        return "CLOUDY"
    if cloud >= 30:
        return "CLOUDY-SUNNY"  
    else:
        return "SUNNY"

