from pickletools import optimize
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
import tensorflow as tf

#extract the data
dataset = pd.read_csv('churn_Modelling.csv')
x = dataset.iloc[:,3 :-1].values
y = dataset.iloc[:,-1]

#encoding the gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

#encoding the geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#build the Ann
ann = tf.keras.models.Sequential()
#hidden and first layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#out put layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#train the ann
ann.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)

#testing my ann
user = ann.predict(sc.transform([[1,0,0,600,1,40,4,60000,2,1,1,50000]])) > .5
print (user)

#show the predictin test results
y_pred = ann.predict(x_test)
y_pred = (y_pred > .5)
print(np.concatenate((y_pred, y_test.values.reshape(-1, 1)), axis=1))

#show the accuracy of the ANN
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

#mark program is done
print("Program Finish")

