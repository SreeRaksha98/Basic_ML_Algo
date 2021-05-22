import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('C:/Users/SREERAKSHA_M_R/Desktop/Random_forest/detection_train.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset.head()
print('-----------------------------------------------------')
dataset.shape
print('-----------------------------------------------------')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
x_train.shape
x_test[:3]

print(y_train)
print('-----------------------------------------------------')

print(y_test)
print('-----------------------------------------------------')

classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
print(classifier.predict(x_test))
print('-----------------------------------------------------')

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
print('-----------------------------------------------------')


