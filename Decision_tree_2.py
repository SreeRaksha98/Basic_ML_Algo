import pandas as pd
import numpy as np
dataset = pd.read_csv('C:/Users/SREERAKSHA_M_R/Desktop/Machine_learning/detection_3.csv')
print('----------------------------------head------------------------------------------------------')
print(dataset.head())

inputs = dataset.drop('Label',axis='columns')
target = dataset['Label']
print('\n----------------------------------Dataset------------------------------------------------------')
print(inputs)
print('\n----------------------------------Target------------------------------------------------------')
print(target)

from sklearn.preprocessing import LabelEncoder
le_Age = LabelEncoder()
le_Natural_Pigmentation = LabelEncoder()
le_Damaged = LabelEncoder()

inputs['Age_n'] = le_Age.fit_transform(inputs['Age'])
inputs['Natural_Pigmentation_n'] = le_Age.fit_transform(inputs['Natural_Pigmentation'])
inputs['Damaged_n'] = le_Age.fit_transform(inputs['Damaged'])
print('\n----------------------------------Modified dataset------------------------------------------------------')
print(inputs.head())

print('\n----------------------------------new dataset------------------------------------------------------')
inputs_n = inputs.drop(['Age','Natural_Pigmentation','Damaged'],axis='columns')
print(inputs_n)
x = inputs_n.iloc[:, :-1].values
y = inputs_n.iloc[:, -1].values
print('\n----------------------------------size-----------------------------------------------------')
print(inputs_n.shape)

print('\n---------------------Splitting the dataset to training and testing set------------------------------------')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
print('\n----------------------------------x_train-----------------------------------------------------')
x_train.shape
print(x_train)
print('\n----------------------------------x_test-----------------------------------------------------')
x_test[:3]
print(x_test)
print('\n----------------------------------y_train-----------------------------------------------------')
print(y_train)
print('\n----------------------------------y_test-----------------------------------------------------')
print(y_test)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,max_depth=3,min_samples_leaf=5)
clf_entropy.fit(x_train, y_train)

print('\n----------------------------------predicting the test set------------------------------------------------------')
y_pred = clf_entropy.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('\n-----------------------------------Confusion matrix-----------------------------------------------------')
print(cm)
print('\n-----------------------------------Accuracy-----------------------------------------------------')
print('Accuracy:',accuracy_score(y_test, y_pred))