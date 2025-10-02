import pandas as pd
import numpy as np
from typing import Union,List
import random

class perceptron:
    #AND GATE PERCEPTRON FOR NOW
    def __init__(self,random_state=42,random_intialize=False):
        self.random_state = random_state
        self.random_intialize = random_intialize
        self.weights = None
        self.bias = 0

    def fit(self,X:pd.DataFrame,activation:str="step"):
        pass

    def predict(self,X:Union[pd.Series,List]):
        pass

    def forward(self,X:np.array):
        pass

    def step_function(self):
        pass
    
    def gradient_descent(self):
        pass

    def weightupdate(self):
        #For step function only
        pass

    def __str__(self):
        pass