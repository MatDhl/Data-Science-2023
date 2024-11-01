# Project:  Implement a random forest classifier for the rice dataset


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

 
data = pd.read_excel('Rice_Cammeo_Osmancik.xlsx')

# Separate the features (X) and the target variable (y)
X = data.drop('Class', axis=1)  # Replace 'target_variable' with the actual column name
y = data['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)


# Confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred, pos_label='Cammeo')  # Specify the positive label
print("Precision:", precision)

# Sensitivity (Recall)
sensitivity = recall_score(y_test, y_pred, pos_label='Cammeo')  # Specify the positive label
print("Sensitivity:", sensitivity)

# F1 Score
f1 = f1_score(y_test, y_pred, pos_label='Cammeo')  # Specify the positive label
print("F1 Score:", f1)
