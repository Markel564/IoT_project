from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
# we import Ann and LSTMPredictorWrapper from python_code
from .ann_model import Ann
from .lstm import LSTMPredictorWrapper
import torch
import torch.multiprocessing as mp
import time 

def get_paths(target_variables, algorithm):

    paths = []

    if algorithm == "ANN":

        for i in range(5):
            paths.append("./models/{}_ann_model.pth".format(target_variables[i]))
    
    else:   
        for i in range(5):
            paths.append("./models/{}_lstm_model.pth".format(target_variables[i]))
            
    return paths



def load_csv_and_wrap_model(args):
    file_path, target_variable, model_path, algorithm = args
    new_df = pd.read_csv(file_path)
    if algorithm == "ANN":
        start_time = time.time()  
        model = Ann(new_df, target_variable=target_variable)
        model.load_state_dict(torch.load(model_path))
        print("Time to load model: ", time.time() - start_time)
    elif algorithm == "LSTM":
        start_time = time.time() 
        model = LSTMPredictorWrapper(new_df, target_variable=target_variable)
        model.load_model(model_path)
        print(f"{algorithm} Time to load model: ", time.time() - start_time)
    return model

def parallel_load_csv_and_wrap_model(args):
    with mp.Pool() as pool:
        models = pool.map(load_csv_and_wrap_model, args)
    return models

    
def get_models(target_variables,algorithm, paths):

    models = []
    args_list = []

    if algorithm == "ANN":
        for i in range(5):
            file_path = "./docs/data/{}/GlobalWeatherRepository_{}.csv".format(target_variables[i], target_variables[i])
            args_list.append((file_path, target_variables[i], paths[i], "ANN"))
    else:
        for i in range(5):
            file_path = "./docs/data/{}/GlobalWeatherRepository_{}.csv".format(target_variables[i], target_variables[i])
            args_list.append((file_path, target_variables[i], paths[i], "LSTM"))

    models = parallel_load_csv_and_wrap_model(args_list)

    return models

