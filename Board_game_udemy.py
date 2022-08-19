# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:40:33 2021

@author: Jagriti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('games.csv')


# Remove any rows without user reviews.
data = data[data["users_rated"] > 0]
# Remove any rows with missing values.
data = data.dropna(axis=0)



features = data.columns.to_list()
features = [i for i in features if i not in ["bayes_average_rating",
                                             "average_rating", "type", 
                                             "name", "id"]]




X = data[features]
Y = data['average_rating']


"""-----------------------------------------------------------------"""
#CORRELATION MATRIX
corrmat= data.corr()
fig = plt.figure(figsize = (10,10))
import seaborn as sns
sns.heatmap(corrmat,vmax=0.8,square= True)


"""-----------------------------------------------------------------"""
#TRAINING THE MODEL
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train, Y_test = train_test_split(X,Y, random_state=5)


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, Y_train)
print('accuracy via linear regression =' ,reg.score(X_test, Y_test))


from sklearn.metrics import mean_squared_error
print('mean square error =',mean_squared_error(reg.predict(X_test), Y_test))





