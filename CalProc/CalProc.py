# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:42:50 2022

@author: tyler.werne
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from scipy import signal
from scipy import integrate
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import simps
from numpy import trapz
import datetime
import os
from matplotlib import pyplot as plt
import string
import sklearn
from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TheilSenRegressor
from sklearn.pipeline import Pipeline
import openpyxl

plt.close('all')


def evaluate_model(X, y, model):
    # define model evaluation method
    cv = model_selection.RepeatedKFold(n_splits=10, n_repeats=3,
                                       random_state=1)
    # evaluate model
    scores = model_selection.cross_val_score(model, X, y,
                                             scoring='neg_mean_absolute_error',
                                             cv=cv, n_jobs=-1)
    # force scores to be positive
    return np.absolute(scores)


calData = r'CalData_03142022.xlsx'

data = pd.read_excel(calData)
Num = np.linspace(1, 13, 12)
df = pd.DataFrame(data,
                  columns=['Mod', '%', 'Sensitivities', 'Length (Inch)', 'Resistance@ 1 kHz', 'Inductance @ 100 kHz'])

Mod = df['Mod'][1:13]
R = df['Resistance@ 1 kHz'][1:13]
K = (10 ** 9) * df['Sensitivities'][1:13]
Length = df['Length (Inch)'][1:13]
L = df['Inductance @ 100 kHz'][1:13]
Range = df['%'][1:13]

X = df[['Length (Inch)', 'Resistance@ 1 kHz', 'Inductance @ 100 kHz']][1:13]
y = (10 ** 9) * df['Sensitivities'][1:13]

'''
Linear Regression
'''

regr = linear_model.LinearRegression()
regr.fit(X, y)
[a, b, c] = regr.coef_
print([a, b, c])
r2 = regr.score(X, y)
print(r2)
results = evaluate_model(X, y, regr)
print('Mean Linear MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))

y_est = regr.predict(X)
plt.figure()
plt.plot(Num, y_est, 'b')
plt.plot(Num, y, 'r')
plt.legend(['Estimate Linear', 'Measured'])

'''
Huber Regression
'''

Huber = HuberRegressor()
# xaxis = np.arange(X.min(), X.max(), 0.01)
Huber.fit(X, y)
[aH, bH, cH] = Huber.coef_
yH = Huber.predict(X)
results = evaluate_model(X, y, Huber)
print('Mean Huber MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))
# plot the line of best fit

plt.figure()
plt.plot(Num, yH, 'b')
plt.plot(Num, y, 'r')
plt.legend(['Estimate Huber', 'Measured'])

'''
Theil Sen Regression
'''
Theil = TheilSenRegressor()
Theil.fit(X, y)
[aT, bT, cT] = Theil.coef_
yT = Theil.predict(X)
results = evaluate_model(X, y, Theil)
print('Mean Theil MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))
# plot the line of best fit

plt.figure()
plt.plot(Num, yT, 'b')
plt.plot(Num, y, 'r')
plt.legend(['Estimate Theil', 'Measured'])

'''
Polynomial Regression
'''

# creating pipeline and fitting it on data
poly2 = PolynomialFeatures(degree=2)
Xt2 = poly2.fit_transform(X)

# poly.fit(X_, y)
# poly_pred = poly.predict(X_)


# generate the regression object
clf2 = linear_model.LinearRegression()
# preform the actual regression
clf2.fit(Xt2, y)
y_Quad = clf2.predict(Xt2)

'''
#sorting predicted values with respect to predictor
sorted_zip = sorted(zip(X_,poly_pred))
x_poly, poly_pred = zip(*sorted_zip)
'''

'''
results = evaluate_model(X_, y, poly)
print('Mean Poly MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))
'''

Poly_coefs = clf2.coef_
print(Poly_coefs)
plt.figure()
plt.plot(Num, y_Quad, 'b')
plt.plot(Num, y, 'r')
plt.legend(['Estimate Quad', 'Measured'])

'''
Cubic
'''

# creating pipeline and fitting it on data
poly3 = PolynomialFeatures(degree=3)
Xt3 = poly3.fit_transform(X)

# poly.fit(X_, y)
# poly_pred = poly.predict(X_)


# generate the regression object
clf3 = linear_model.LinearRegression()
# preform the actual regression
clf3.fit(Xt3, y)
y_cube = clf3.predict(Xt3)

'''
#sorting predicted values with respect to predictor
sorted_zip = sorted(zip(X_,poly_pred))
x_poly, poly_pred = zip(*sorted_zip)
'''

'''
results = evaluate_model(X_, y, poly)
print('Mean Poly MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))
'''

Poly_coefs = clf3.coef_
print(Poly_coefs)
plt.figure()
plt.plot(Num, y_cube, 'b')
plt.plot(Num, y, 'r')
plt.legend(['Estimate Cube', 'Measured'])

plt.show()
