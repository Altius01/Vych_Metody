# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv", sep='\t')
x_1 = data.x1.values
y_1 = data.y1.values

x_2 = data.x2.values
y_2 = data.y2.values

x_3 = data.x3.values
y_3 = data.y3.values

x_4 = data.x4.values
y_4 = data.y4.values

reg = LinearRegression()

x_1 = x_1.reshape(-1, 1)
#y_1 = y_1.reshape(1, -1)

reg = reg.fit(x_1,y_1)

plt.subplot(2,2,1)
plt.scatter(x_1[:,0], y_1)
plt.plot(x_1[:,0], reg.predict(x_1))

x_2 = x_2.reshape(-1, 1)
#y_1 = y_1.reshape(1, -1)

reg = reg.fit(x_2,y_2)

plt.subplot(2,2,2)
plt.scatter(x_2[:,0], y_2)
plt.plot(x_2[:,0], reg.predict(x_2))

x_3 = x_3.reshape(-1, 1)
#y_1 = y_1.reshape(1, -1)

reg = reg.fit(x_3,y_3)

plt.subplot(2,2,3)
plt.scatter(x_3[:,0], y_3)
plt.plot(x_3[:,0], reg.predict(x_3))

x_4 = x_4.reshape(-1, 1)
#y_1 = y_1.reshape(1, -1)

reg = reg.fit(x_4,y_4)

plt.subplot(2,2,4)
plt.scatter(x_4[:,0], y_4)
plt.plot(x_4[:,0], reg.predict(x_4))


