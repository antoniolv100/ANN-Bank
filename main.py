import numpy as np
import pandas as pd
import tensorflow as tf

#extract the data
dataset = pd.read_csv('churn_Modelling.csv')
x = dataset.iloc[:,3 :-1].values
y = dataset.iloc[:,-1]


print("done")

