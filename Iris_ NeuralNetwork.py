######  Using scikit learn to implement a neural network for the Iris dataset

#Import necessary libraries to load data:

import numpy as np
import pandas as pd

data = pd.read_csv('Iris.csv',delimiter=';', encoding='utf-8')
view = data.head()
print (view)

#arrange data into an array of features, X and target, t :
data = np.genfromtxt('Iris.csv', delimiter=';', dtype=str, encoding='utf-8-sig')
#data = np.loadtxt('Iris.csv', delimiter=';', dtype=str, encoding='utf-8-sig')


X = data[:, 0:4]    
t = data[:, 4]    #5th colum in csv is the t values

from sklearn.preprocessing import OneHotEncoder

# Reshape the target array to a 2-dimensional array
t_reshaped = np.reshape(t, (-1, 1))
#X = np.reshape(X, (-1, 1))

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the reshaped array
t= encoder.fit_transform(t_reshaped)

print("X:", X)
print("t:", t)


#Train the model using 80% of the data:
from sklearn.model_selection import train_test_split

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)

#scaling the data using min-max normaliztion

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


##Creating the N-network:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

#Split the dataset into training and validation sets
X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2, random_state=42)

#Normalize the data using min-max normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("X_train_scaled shape:", X_train_scaled.shape)


#Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(20, activation='sigmoid', input_shape=(4,)),  #4 neurons in input
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(3, activation='softmax')
])

#Compile the model with sum-of-squares loss
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the neural network
epochs = 100
batch_size = 35

for epoch in range(epochs):
    # Perform one epoch of training
    model.fit(X_train_scaled, t_train, batch_size=batch_size, epochs=1, verbose=0)
    
    # Calculate the sum-of-squares loss on the training set
    train_loss = np.mean(np.square(model.predict(X_train_scaled) - t_train))
    
    # Calculate the sum-of-squares loss on the validation set
    val_loss = np.mean(np.square(model.predict(X_val_scaled) - t_val))
    
    # Print the losses after each epoch
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# Evaluate the model on the test set

predictions = model.predict(X_test_scaled)
test_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(t_test, axis=1))

print(f"Test Accuracy = {test_accuracy:.4f}")

