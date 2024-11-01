###Import scikitlearn

import  sklearn
import pandas as pd

###Load dataset

data = pd.read_csv('Percept1.csv', sep=';')

d = data.head()         #data preview
print(d)


###Use scikit learn in the algorithm
##Separate the data to form two matrices, X and and t:

import numpy as np

# Load the CSV file
data = np.loadtxt('Percept1.csv', delimiter=';')

X = data[:, :2]     #first 2 columns form x1 and x2

t = data[:, 2:].ravel()    #third colum in csv is the t values

#print("X:", X)
#print("t:", t)

###Initialize the perceptron model
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

#Train the model using 80% of the data:
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)

model = Perceptron(eta0=0.1, max_iter=100)  #Set the learming rate and iterations to 0.1 and 100 respectively

###Train the perceptron
model.fit(X_train, t_train)

#Testing the perceptron on the remaining data
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = np.mean(y_pred == t_test)
print("The accuracy is:", accuracy)

### Evaluate the weights of the perceptron

weights = model.coef_
print("Weights:", weights.round(2))

### Make predictions on new data
new_data = np.array([[1.8, 2.2], [0.4, 1.2]])
predictions = model.predict(new_data)

print("The prediction to the new input values are:",  predictions)
