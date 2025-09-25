import pandas as pd
from typing import Union,List
import math
#WITHOUT PARRELELLIZATION FOR NOW
class id3:
    
    def __init__(self,data,target_column:str):
        pass
        self.data = pd.DataFrame(data)
        self.target = self.data[target_column]
    
    def one_class_entropy(self,x):
        #X is in the form pd.Series
        pass
    
    def split_data(self,splitting_class,splitting_value:Union[int,str,list,float]):
        #Splits into 2 parts for now
        pass

    def fit(self,X):
        #Recursion logic
        pass

    def predict(self,X):
        pass