"""
This file contains a function that packs the information in an array to be sent to the front end
"""

def pack_info(condition_1, condition_2, condition_3, temp_day_1, temp_day_2, temp_day_3, humidity_day_1, precip_day_1, wind_day_1):
    """
    This functions packs the information in an array to be sent to the front end

    The structure of the array will be as follows:

    [condition_1, condition_2, condition_3, temp_day_1, temp_day_2, temp_day_3,
    humidity_day_1, precip_day_1. wind_day_1]
    
    """
    
    info_to_send = []

    # append all the information into an array
    info_to_send.append(condition_1)
    info_to_send.append(condition_2)
    info_to_send.append(condition_3)


    info_to_send.append(round(temp_day_1, 0)) # round the temperature to 0 decimals
    info_to_send.append(round(temp_day_2, 0))
    info_to_send.append(round(temp_day_3, 0))

    # and also round the humidity, precip and wind to respectively 0, 2 and 1 decimals
    info_to_send.append(round(humidity_day_1, 0)) 
    info_to_send.append(round(precip_day_1, 2))
    info_to_send.append(round(wind_day_1, 1))

    return info_to_send