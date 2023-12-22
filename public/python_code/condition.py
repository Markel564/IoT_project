"""
This file receives the 5 target variables and determines the weather condition
"""



def condition(temp, precip, cloud, humidity, wind):
    
    """
    options are:
    - sunny
    - cloudy            
    - rainy             
    - snowy             
    - windy             
    - cloudy-sunny      
    """
    if wind >= 10:
        return "windy"  
    if precip >= 1 and temp >= 0:
        return "rainy" 
    if temp <= 0 and precip >= 1:
        return "snowy"
    if cloud >= 70: # precip is low so it does not rain
        return "cloudy"
    if cloud >= 50:
        return "cloudy-sunny"     
    else:
        return "sunny"