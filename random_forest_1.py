import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/SREERAKSHA_M_R/Desktop/Machine_learning/detection_1.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('\n----------------------------------head()------------------------------------------------------')
print(dataset.head())
print('\n----------------------------------size()-----------------------------------------------------')
print(dataset.shape)
print('\n----------------------------------info()-----------------------------------------------------')
print(dataset.info)

print('\n---------------------Splitting the dataset to training and testing set--------------------------')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
print('\n----------------------------------x_train()-----------------------------------------------------')
x_train.shape
print(x_train)
print('\n----------------------------------x_test()-----------------------------------------------------')
x_test[:3]
print(x_test)
print('\n----------------------------------y_train()-----------------------------------------------------')
print(y_train)
print('\n----------------------------------y_test()-----------------------------------------------------')
print(y_test)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
print('\n----------------------------------predicting the test set------------------------------------------------------')
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('\n-----------------------------------Confusion matrix()-----------------------------------------------------')
print(cm)
print('\n-----------------------------------Accuracy-----------------------------------------------------')
print('Accuracy:',accuracy_score(y_test, y_pred))