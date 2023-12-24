"""
This file has the functions related to the condition of the weather

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
    if wind >= 10:
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
        return "sunny"

