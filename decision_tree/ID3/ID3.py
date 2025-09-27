import pandas as pd
from typing import Union,List,Tuple
import math

#WITHOUT PARRELELLIZATION FOR NOW
class id3:
    
    def __init__(self,data,target_column:str,tree_depth:Union[int,None]=None):
        self.data = pd.DataFrame(data)
        self.target_column = target_column
        self.target = self.data[target_column]
        self.columns = [column for column in self.data if column != self.target]
        self.target_value = list(set(self.target.tolist()))
        self.main_tree = []
        
    def entropy(self,x:pd.Series):
        #X is in the form pd.Series
        Entropy = 0.0
        class_counts = []
        totals = len(x)
        for i in range(len(self.target_value)):
            temp = 0
            for j in range(len(x)):
                if x.iloc[j] == self.target_value[i]:
                    temp+=1
            class_counts.append(temp)
        for i in range(len(class_counts)):
            value = class_counts[i]
            if value > 0:
                p = value/totals
                Entropy = Entropy - (p * math.log2(p))
        return Entropy
    
    def information_gain(self,feature,X=None):
        #X is the total data we have at this stage. Feature is the feature we want to get the IG.
        if X is None:
            X = self.data
        totals = len(X)
        parent_Entropy = self.entropy(X[self.target_column])
        children = X[feature].unique()
        child_Entropy = 0
        for i in children:
            X_child = X.loc[X[feature]==i]
            Y_child = X_child[self.target_column]
            child_Entropy +=  (len(X_child)/totals)*(self.entropy(Y_child))
        gain = parent_Entropy - child_Entropy
        return gain

    def fit(self,X):
        #Tree building logic here
        pass

    def predict(self,X):
        pass