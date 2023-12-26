"""
This file contains two functions that returns the current time in a string format.

The string format returned by the first function is:

Day of the week, Month Day, Name of Month, Year

The string format returned by the second function is:
hh:mm

"""

import datetime

def get_date():
    """
    This function returns the current date in the following format:

    Day of the week, Month Day, Name of Month, Year

    :return: date in string format
    """
    
    date = datetime.datetime.now()
    day = date.strftime("%A")
    month = date.strftime("%B")
    day_of_month = date.strftime("%d")
    year = date.strftime("%Y")
    return day + ", " + month + " " + day_of_month + ", " + year


def get_hour():
    """
    This function returns the current hour in the following format:

    hh:mm
    :return: hour in string format
    """

    date = datetime.datetime.now()
    hour = date.strftime("%H")
    minute = date.strftime("%M")
    return hour + ":" + minute


def get_dates():
    """
    This function returns an array with this day and the following 2 days

    for example, if it was monday, it would return [monday, tuesday, wednesday]
    """

    date = datetime.datetime.now()
    day = date.strftime("%A")
    days = [day]
    for i in range(1, 3):
        date += datetime.timedelta(days=1)
        day = date.strftime("%A")
        days.append(day)
    return days