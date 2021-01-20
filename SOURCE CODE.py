# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:34:13 2021

@author: Haran
"""


import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt


data = pd.read_csv("scores.csv")  
print("Data read!")
data.head()

data.plot(x='Hours', y='Scores', style='*')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

x = data.iloc[:,:1].values    
y = data.iloc[:, 1].values  

print("x= ",x)
print("y= ",y)


from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)  

from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(x_train, y_train)   

line = regressor.coef_*x+regressor.intercept_  
plt.scatter(x,y)  
plt.plot(x, line);  
plt.show()  

hours=0
while hours!=[[-1]]:
    hours = [[float(input("Enter number of hours studied(Enter '-1' to stop):"))]]
    if hours == [[-1]]:
        break
    pred = regressor.predict(hours)
    if pred[0]>100:
        pred[0]=100
    print("Number of hours = {}".format(hours[0][0]))
    print("Prediction Score = {}".format(pred[0]))
