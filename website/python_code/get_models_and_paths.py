"""
This file contains functions that are used to load the models and paths for the models.

It also uses threads because loading the models takes a lot of time, so we load them in parallel,
which reduces the time considerably.
"""

from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .ann_model import Ann
from .lstm import LSTMPredictorWrapper
import torch
import torch.multiprocessing as mp
import time 

def get_paths(target_variables, algorithm):
    """
    This function returns a list of paths for the models according to the target variables and the algorithm
    """
    paths = []

    if algorithm == "ANN":

        for i in range(5):
            paths.append("./models/{}_ann_model.pth".format(target_variables[i]))
    
    else:   
        for i in range(5):
            paths.append("./models/{}_lstm_model.pth".format(target_variables[i]))
            
    return paths



def load_csv_and_wrap_model(args):
    """
    This function loads model 

    """

    file_path, target_variable, model_path, algorithm = args # unpack the arguments
    new_df = pd.read_csv(file_path) # load the csv
    if algorithm == "ANN": 
        start_time = time.time() # start a timer (for debugging purposes)  
        model = Ann(new_df, target_variable=target_variable) #create the model's architecture
        model.load_state_dict(torch.load(model_path)) # load the model's weights
        print("Time to load model: ", time.time() - start_time)

    elif algorithm == "LSTM":
        start_time = time.time() 
        model = LSTMPredictorWrapper(new_df, target_variable=target_variable)
        model.load_model(model_path)
        print(f"{algorithm} Time to load model: ", time.time() - start_time)

    return model

def parallel_load_csv_and_wrap_model(args):
    """
    This function calls the function above in parallel
    """
    with mp.Pool() as pool:
        models = pool.map(load_csv_and_wrap_model, args)
    return models

    
def get_models(target_variables,algorithm, paths):
    """
    This function returns a list of models according to the target variables and the algorithm
    It uses the functions above to load the models
    """
    models = []
    args_list = []

    if algorithm == "ANN": # if the algorithm is ANN, we load the ANN models
        for i in range(5):
            # get the file path
            file_path = "./docs/data/{}/GlobalWeatherRepository_{}.csv".format(target_variables[i], target_variables[i])
            # append the arguments to the args_list
            args_list.append((file_path, target_variables[i], paths[i], "ANN"))
    else:
        for i in range(5):
            # get the file path
            file_path = "./docs/data/{}/GlobalWeatherRepository_{}.csv".format(target_variables[i], target_variables[i])
            # and append the arguments to the args_list
            args_list.append((file_path, target_variables[i], paths[i], "LSTM"))

    models = parallel_load_csv_and_wrap_model(args_list)

    return models

