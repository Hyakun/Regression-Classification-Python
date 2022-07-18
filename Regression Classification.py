# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:48:51 2022

@author: xdany
"""


import pandas as pd
import seaborn as sns

dataset = pd.read_csv('\\USA_Housing.csv')

dataset.columns = ['AI', 'AHA', 'ANR', 'ANB', 'AP', 'PRICE']


x1 = dataset[['AI', 'AHA', 'ANR', 'ANB', 'AP']]
y1 = dataset[['PRICE']]


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()

regressor1.fit(x1_train, y1_train)


print('Intercept: \n', regressor1.intercept_)
print('Coefficients: \n', regressor1.coef_)



y1_pred = regressor1.predict(x1_test)

from sklearn.metrics import r2_score
print('Coefficient of determination (R^2): %.2f' % r2_score(y1_test, y1_pred))

print(y1_pred)



sns.regplot(y1_test, y1_pred)




import pandas as pd

dataset = pd.read_csv('\\PhoneClass.csv')

dataset.info()
dataset.describe()
dataset.head()

dataset.columns = ['BP', 'B', 'CS', 'DS', 'FC', 'FG', 'IM', 'MD', 'MW', 'NC', 'PC', 'PH', 'PW', 'RAM', 'SCH', 'SCW', 'TT','TG', 'TS','WIFI', 'PR']


x1 = dataset[['BP', 'B', 'CS', 'DS', 'FC', 'FG', 'IM', 'MD', 'MW', 'NC', 'PC', 'PH', 'PW', 'RAM', 'SCH', 'SCW', 'TT','TG', 'TS','WIFI']]
y1 = dataset[['PR']]


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2, random_state = 0)



from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier(n_estimators=100)

rfc1.fit(x1_train, y1_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)

predictions1 = rfc1.predict(x1_test)


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y1_test, predictions1))
print(confusion_matrix(y1_test, predictions1))

import matplotlib.pyplot as plt
import seaborn as sn

plt.figure(figsize=(10,7))
sn.heatmap(confusion_matrix(y1_test, predictions1), annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


print(rfc1.score(x1_test, y1_test))





















