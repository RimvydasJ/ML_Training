import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, model_selection, neighbors


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True) #For the data we don't know we treat them as outliers. ? is not known data
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test,y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2) #How much of the data should be generated for testing (test_size)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,3,4,2,3,2,5]])
print(clf.predict(example_measures))