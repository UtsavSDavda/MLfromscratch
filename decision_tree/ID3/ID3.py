import pandas as pd
from typing import Union
import math

#WITHOUT PARRELELLIZATION FOR NOW
class Node:
    def __init__(self,label=None,feature=None,children:dict = {},isroot:bool=False,isleaf:bool=False):
        self.feature = feature
        self.children = children
        self.label = label
        self.isroot = isroot
        self.isleaf = isleaf

class id3:
    def __init__(self,data,target_column:str,tree_depth:Union[int,None]=None):
        self.data = pd.DataFrame(data)
        self.target_column = target_column
        self.target = self.data[target_column]
        self.columns = [column for column in self.data if column != self.target_column]
        self.target_value = list(set(self.target.tolist()))
        self.main_tree = Node(isroot=True)
        if tree_depth is not None:
            self.tree_depth = tree_depth
        else:
            self.tree_depth = len(self.data[self.columns[0]])
    
    def entropy(self,x:pd.Series):
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
    
    def fit(self):
        #Tree building logic here
        self.root = self.assign(self.data, depth=self.tree_depth, featurelist=self.columns)
    
    def assign(self,X:pd.DataFrame,depth=0,featurelist:list=[])->Node:
        #Adds a node
        if X[self.target_column].nunique() == 1:
            return Node(label=X[self.target_column].iloc[0],isleaf=True)
        if depth > self.tree_depth or len(featurelist) == 0:
            return Node(label=X[self.target_column].mode()[0],isleaf=True)
        gainmax = 0
        featuremax = None
        for f in self.columns:
            #Default init as 1st column
            if featuremax is None:
                featuremax = f
            #Loop to find the best
            feature_gain = self.information_gain(f,X)
            if feature_gain > gainmax:
                featuremax = f
                gainmax = feature_gain
        internal_node = Node(label=X[self.target_column].mode()[0],feature=featuremax)
        featurelist.remove(featuremax)
        for child in X[featuremax].unique():
            subset = X.loc[X[featuremax]==child]
            childnode = self.assign(X=subset,depth=depth+1,featurelist=featurelist)
            internal_node.children[child] = childnode
        return internal_node
    
    def predict(self,X:pd.DataFrame)->list:
        return X.apply(self.predict_single, axis=1).tolist()
    
    def predict_single(self,X,current_node=None):
        if current_node is None:
            current_node = self.root
        if self.root is None:
            #If tree dosen't exist, just to not break code
            return self.data[self.columns[0]].iloc[0]
        if current_node.isleaf or len(current_node.children) == 0:
            return current_node.label
        nextnode = None
        featurevalue = X[current_node.feature]
        if featurevalue in current_node.children:
            nextnode = current_node.children[featurevalue]
            return self.predict_single(X,current_node=nextnode)
        return current_node.label