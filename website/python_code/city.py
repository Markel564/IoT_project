"""
This file returns the city + the country of the city
"""


def get_city_and_country(city):
    
    if city == "Paris":
        return "Paris, France".upper()
    elif city == "Madrid":
        return "Madrid, Spain".upper()
    elif city == "Stockholm":
        return "Stockholm, Sweden".upper()
    elif city == "Tokyo":
        return "Tokyo, Japan".upper()