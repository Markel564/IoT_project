"""
This file contains two functions that returns the current time in a string format.

The string format returned by the first function is:

Day of the week, Month Day, Name of Month, Year

The string format returned by the second function is:
hh:mm

"""

import datetime

def get_date(day):
    """
    This function returns the current date in the following format:

    Day of the week, Month Day, Name of Month, Year

    :return: date in string format
    """
    

    # get the GMT date
    date = datetime.datetime.now(datetime.timezone.utc)
    if day == 0:
        day = date.strftime("%A") # get the day of the week
        month = date.strftime("%B") # get the name of the month
        day_of_month = date.strftime("%d") # get the day of the month
        year = date.strftime("%Y") # get the year

    if day == 1: # this means it is actually one more (for example, we calculated the time of Tokyo and it is the next day rather than GMT)
        date += datetime.timedelta(days=1)
        day = date.strftime("%A")
        month = date.strftime("%B")
        day_of_month = date.strftime("%d")
        year = date.strftime("%Y")
    
    elif day == -1:
        date -= datetime.timedelta(days=1)
        day = date.strftime("%A")
        month = date.strftime("%B")
        day_of_month = date.strftime("%d")
        year = date.strftime("%Y")

    return day + ", " + month + " " + day_of_month + ", " + year # return the date in the format specified above


def get_hour(city):
    """
    This function returns the current hour in the following format:

    hh:mm
    :return: hour in string format and a day (0, 1 or -1) depending if the hour is negative or positive (for later use)

    the cities can either be "Buenos Aires", "Madrid", "Windhoek" or "Tokyo"
    """
    
    # GMT time
    date = datetime.datetime.now(datetime.timezone.utc)
    hour = date.hour # get the hour
    minute = date.minute # get the minute

    day = 0 # we set the day to 0
    # now we need to add the offset depending on the city
    if city == "Buenos Aires":
        hour -= 3
        # if the hour is negative, we need to add 24 to it
        if hour < 0:
            day = -1 # we substract one day 
            hour += 24
    elif city == "Madrid":
        hour += 1
        if hour > 23:
            day = 1
            hour -= 24
    elif city == "Windhoek":
        hour += 2
        if hour > 23:
            day = 1
            hour -= 24
    elif city == "Tokyo":
        hour += 9
        if hour > 23:
            day = 1
            hour -= 24
    else:
        return "Error"
    
    hour = str(hour)
    minute = str(minute)
    return [day, hour + ":" + minute]


def get_dates(day_param):
    """
    This function returns an array with this day and the following 2 days

    for example, if it was monday, it would return [monday, tuesday, wednesday]

    day_param can be 0, 1 or -1, indicating if we have to substract one day or add one day to the current day
    since the first calculation is done with GMT time, and a city could be in a different day
    """

    date = datetime.datetime.now()
    day = date.strftime("%A")
    print ("Day: ", day)
    days = [day]
    if day_param == 0:
        for i in range(1, 3):
            date += datetime.timedelta(days=1)
            day = date.strftime("%A")
            days.append(day)
            print (day)
    # if the day is 1, it means that it is actually one more
    elif day_param == 1:
        for i in range(1, 4):
            date += datetime.timedelta(days=1)
            day = date.strftime("%A")
            days.append(day)
        days = days[1:]
    else: # day is -1
        
        days = [] # since the array contains today, we have to eliminate it to start from yesterday
        date -= datetime.timedelta(days=1) #we substract one day
        day = date.strftime("%A")
        days.append(day)
        date += datetime.timedelta(days=1) 
        day = date.strftime("%A")
        days.append(day)
        date += datetime.timedelta(days=1) 
        day = date.strftime("%A")
        days.append(day)

    return days