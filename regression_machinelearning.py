import pandas as pd
from pandas.core.frame import DataFrame
import quandl as q #Library to get raw data from internet.
import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import os


style.use('ggplot')
classifierFile = 'linearregression.pickle'
datafile = 'googlestock.pickle'

#Very important to have meaningful data, that impacts it.
#Features defines your data based on which you going to learn
#Labels defines based on what you are going predict(Result).

df = DataFrame()
if os.path.isfile(datafile):
    data_in = open(datafile, 'rb')
    df = pickle.load(data_in)
else:
    df = q.get('WIKI/GOOGL')
    df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
    df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume',]]
    with open(datafile, 'wb') as f:
        pickle.dump(df, f)


forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #We are filling NaN data. Some data will be missing, so instead of getting rid of it, we fill with outlier

forecast_out = int(math.ceil(0.1*len(df))) #We will predict only 10% of the data in the future
#print(forecast_out) #Result when testing 35 days

df['label'] = df[forecast_col].shift(-forecast_out) #"inserts" number of rows in the beggining

X = np.array(df.drop(['label'],1)) #Features
X = preprocessing.scale(X) #Feature scaling is a technique to standardize the features
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label']) #Label

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) #data shuffling

clf = LinearRegression() #n_jobs will define how many threads to process
#clf = svm.SVR(kernel='poly') support vector regression, different algorithm
if os.path.isfile(classifierFile):
    pickle_in = open(classifierFile, 'rb')
    clf = pickle.load(pickle_in)
else:
    clf.fit(X_train, y_train)
    with open(classifierFile, 'wb') as f:
        pickle.dump(clf, f)

accuracy = clf.score(X_test,y_test)

#print(accuracy) # The result i got 0.9792519628373169. This means that 98% accuracy for 35 upcoming days

forecast_set = clf.predict(X_lately)
#print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()