"""
Tiffany Berntsen - for Verus Research Rogowski Coil calibration for the data
V1, the Rogowski coil data is plotted and integrated, 
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

from numpy import diff
import scipy.special as special
from scipy.integrate import simps
from numpy import trapz
import datetime
from ProcessingFunctions import *
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
from sklearn.linear_model import Ridge
import openpyxl
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# plot settings
plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'lines.linewidth': 5})

# Global variables
sensitivity = 1  # 67.6301041736064e-9#0.217*1e-9; #V/(A/s)
sense_array = []
sense_array2 = []
sense_array3 = []
sense_array4 = []
pear_array = []
int_array = []
int_array2 = []
int_array3 = []
int_array4 = []
pass_array = []

count = 0
pear_sense = 0.1  # pearson sensativity is 0.1 V/A
att_fact_pear = 40.0  # attinuation factor of pearson
load = 2  # dependant on what the pearson is expecting
rise_time = 5.6 * 1e-6  # typical rise time of signal
R = 50  # resistance in calibration system

# Oscope scaling
pscale = 0.5
rvscale = 2
rascale = 0.05

# import csv files from folder
path = os.getcwd()  # currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"))  # files in directory

# Checksum function
checksum(csv_files, path)
# End of checksum function

# loop over the list of csv files
pass_rogo = []
rogo = []
pear_curr = []
plt.close('all')

for file in csv_files:

    # Variables for each file
    time = []
    pear = []  # pearson voltage
    pear_curr = []  # pearson current
    rogo = []  # raw rogowski voltage
    rogo_vol = []  # integrated raw rogowski voltage
    rogo_curr = []  # rogowski current
    rogo_curr2 = []  # rogowski current
    rogo_curr3 = []  # rogowski current
    rogo_curr4 = []  # rogowski current
    pass_rogo = []  # passivly integrated rogowski current
    pass_curr = []  # current of passive rogowski

    # read the csv file
    data = pd.read_csv(file)  # , header=14)
    count = count + 1
    print(file)

    # setting variables to channels and scale to base units
    df = pd.DataFrame(data, columns=['TIME', 'CH1', 'CH2', 'CH3', 'CH4'])
    time = df['TIME']  # seconds
    Fs = 1000e6
    time_fix = np.linspace(time[0], time[time.size - 1],
                           num=int(time.size))

    # --------Pearson Processing----------
    pear = df['CH1']  # attenuated voltage
    pear_att = att(pear, att_fact_pear)  # voltage
    for x in range(len(pear_att)):  # amperage
        # can increase speed by pre-allocating pear_curr, multiplied by 2 due to Pearson impedances
        pear_curr.append(pear_att[x] * 2 / pear_sense)
    # pear_peak = find_peak(pear_curr)
    # print('Pearson current = %0.2f A' % pear_peak)
    # pear_array.append(pear_peak)

    # --------Rogowski Voltage Processing----------

    rogo2 = df['CH2'];  # K*dI/dt
    '''
    for x in range(1, len(rogo2)):
        if(rogo2[x] > 0.001):
            rogo2[x] = rogo2[x-1]
    if(rogo2[0] > 0.001):
        rogo2[0] = rogo2[1]
    '''
    int_rogo2 = integration(rogo2, 1, time_fix, 1)  # K*I
    for x in range(len(int_rogo2)):
        rogo_curr2.append(int_rogo2[x] / 4)  # divided by 4 loops
    # int_peak2 = find_peak(rogo_curr2)                      #peak of integrated current
    # print('Integrated Rogwoski sensitivity current = %0.10f V*s' % int_peak2)
    # int_array2.append(int_peak2)

    '''
    th = rogo_curr2 > (0.70*max(rogo_curr2))
    th[1:][th[:-1] & th[1:]] = False
    occurrences_of_true = np.where(th == True)
    start = occurrences_of_true[0][0]
    '''
    # start = start[0]
    PFa = 0.000001
    refLength = 50
    guardLength = 500

    CFARThreshold = np.abs(CFAR_SS(rogo2, PFa, refLength, guardLength))

    for t in range(600, len(CFARThreshold)):
        if ((rogo2[t] > CFARThreshold[t]) or (rogo2[t] < -1 * CFARThreshold[t])):
            start = t
            break;

    # start = start[0]
    th = np.abs(rogo_curr2) > (0.40 * max(np.abs(rogo_curr2)))
    th[1:][th[:-1] & th[1:]] = False
    th = np.flip(th)

    occurrences_of_true = np.where(th == True)

    if (isinstance(occurrences_of_true, int)):
        end_rev = occurrences_of_true
    else:
        end_rev = occurrences_of_true[0][0]
        if (end_rev == 0):
            end_rev = occurrences_of_true[0][1]
    end = len(rogo_curr2) - end_rev
    # end = end_rev
    # end = end[0]

    x_peaks2 = [x for x in range(start, end)]
    int_peaks2 = rogo_curr2[start:end]
    pear_peaks2 = pear_curr[start:end]
    print('Integrated Rogwoski sensitivity current = %0.10f V*s' %
          np.mean(int_peaks2))

    # end = end[0]

    '''
    #x_peaks = [enumerate(rogo_curr)]
    y_peaks = rogo_curr[x_peaks[0]:x_peaks[-1]]
    start = x_peaks[0]
    end = x_peaks[-1]
    '''
    '''
    x_peaks2 = [x for x in range(start,end)]
    int_peaks2 = rogo_curr2[start:end]
    pear_peaks2 = pear_curr[start:end]
    print('Integrated Rogwoski sensitivity current = %0.10f V*s' % 
          np.mean(int_peak2))
    int_array.append(int_peak2)
    '''
    '''
    rogo3 = df['CH3']  # K*dI/dt
    for x in range(1, len(rogo3)):
        if(rogo3[x] > 0.001):
            rogo3[x] = rogo3[x-1]
    int_rogo3 = integration(rogo3, 1, time_fix, 1)  # K*I
    for x in range(len(int_rogo3)):
        rogo_curr3.append(int_rogo3[x]/4)  # divided by 4 loops
    int_peak3 = find_peak(rogo_curr3)  # peak of integrated current
    print('Integrated Rogwoski sensitivity current = %0.10f V*s' % int_peak3)
    int_array3.append(int_peak3)
    '''
    '''
    th = rogo_curr3 > (0.60*max(rogo_curr3))
    th[1:][th[:-1] & th[1:]] = False
    occurrences_of_true = np.where(th == True)
    start = occurrences_of_true[0][0]
    '''

    '''
    PFa = 0.0001
    refLength = 100
    guardLength = 5
    
    CFARThreshold = CFAR(rogo3, PFa, refLength, guardLength)

    for t in range(0,len(CFARThreshold)):
        if ((rogo3[t] > CFARThreshold[t]) or (rogo3[t] < -1*CFARThreshold[t])):
            start = t
            break;
    
    #start = start[0]
    th = rogo_curr3 > (0.60*max(rogo_curr3))
    th[1:][th[:-1] & th[1:]] = False
    th = np.flip(th)

    occurrences_of_true = np.where(th == True)
    end_rev = occurrences_of_true[0][0]
    end = len(rogo_curr3) - end_rev
    #end = end_rev
    #end = end[0]
    '''
    '''
    #x_peaks = [enumerate(rogo_curr)]
    y_peaks = rogo_curr[x_peaks[0]:x_peaks[-1]]
    start = x_peaks[0]
    end = x_peaks[-1]
    '''
    '''
    x_peaks3 = [x for x in range(start, end)]
    int_peaks3 = rogo_curr3[start:end]
    pear_peaks3 = pear_curr[start:end]
    print('Integrated Rogwoski sensitivity current = %0.10f V*s' %
          np.mean(int_peak3))
    int_array.append(int_peak3)
    '''
    '''
    rogo4 = df['CH4'];                                    #K*dI/dt
    int_rogo4 = integration(rogo4, 1, time_fix, 1)         #K*I
    for x in range(len(int_rogo4)):
        rogo_curr4.append(int_rogo4[x]/4)                  #divided by 4 loops
    int_peak4 = find_peak(rogo_curr4)                      #peak of integrated current
    print('Integrated Rogwoski sensitivity current = %0.10f V*s' % int_peak4)
    int_array4.append(int_peak4)
    
    th = rogo_curr4 > (0.70*max(rogo_curr4))
    th[1:][th[:-1] & th[1:]] = False
    occurrences_of_true = np.where(th == True)
    start = occurrences_of_true[0][0]
    #start = start[0]
    th = rogo_curr4 > (0.7*max(rogo_curr4))
    th = np.flip(th)
    th[1:][th[:-1] & th[1:]] = False
    occurrences_of_true = np.where(th == True)
    end_rev = occurrences_of_true[0][1]
    end = len(rogo_curr4) - end_rev
        
        
        #end = end[0]
    '''
    '''
    #x_peaks = [enumerate(rogo_curr)]
    y_peaks = rogo_curr[x_peaks[0]:x_peaks[-1]]
    start = x_peaks[0]
    end = x_peaks[-1]
    '''
    '''
    x_peaks4 = [x for x in range(start,end)]
    int_peaks4 = rogo_curr4[start:end]
    pear_peaks4 = pear_curr[start:end]
    print('Integrated Rogwoski sensitivity current = %0.10f V*s' % 
          np.mean(int_peak4))
    '''
    # plt.plot(time, int_rogo2)
    # , time, int_rogo3, time, int_rogo4)
    # ---------Sensitivity Calcualtions-----------
    # sensitivity2 = (int_peak2/pear_peak)              #V*s/A
    # sensitivity3 = (int_peak3/pear_peak)  # V*s/A
    # sensitivity4 = (int_peak4/pear_peak)              #V*s/A
    # print('Calculated Sensitivity CH2 = %0.12f V*s/A' % sensitivity2)
    # print('Calculated Sensitivity CH3 = %0.12f V*s/A' % sensitivity3)
    # print('Calculated Sensitivity CH4 = %0.12f V*s/A' % sensitivity4)

    sensitivities2 = int_peaks2
    for x in range(0, len(int_peaks2)):
        sensitivities2[x] = int_peaks2[x] / (pear_peaks2[x] / (0.1))  # V*s/A
    sensitivity2 = np.mean(sensitivities2)
    time_scale = time_fix[1] - time_fix[0]
    
    '''
    for x in range(0, len(deriv)):
        if (np.isinf(deriv[x])):
            deriv[x] = np.mean(diff(rogo_curr2[start + x:start + x + 2]) / time_scale)
    deriv = deriv[start:end]
    '''
    X = np.asarray(rogo_curr2[start:end])

    if (isinstance(X, list) == True):
        print('Is list')
        X = np.concatenate(X, axis=1)
    else:
        print('IS not list')
    X = np.reshape(X, (-1, 1))
    I = np.reshape(np.argsort(X.ravel()),(-1,1))
    # deriv= np.append(deriv, deriv[-1])
    
    th = rogo2[start:end] > (0)
    th[1:][th[:-1] & th[1:]] = False
    #th = np.flip(th)

    occurrences_of_true = np.where(th == True)
    
    rogo_cat = rogo2[start:end]
    for i in range(0,len(occurrences_of_true)):
        if(occurrences_of_true[i] == True):
            rogo_cat[i] = 1
    
    th = rogo2[start:end] == (0)
    th[1:][th[:-1] & th[1:]] = False
    #th = np.flip(th)

    occurrences_of_true = np.where(th == True)
    
    for i in range(0,len(occurrences_of_true)):
        if(occurrences_of_true[i] == True):
            rogo_cat[i] = 0
    
    th = rogo2[start:end] < (0)
    th[1:][th[:-1] & th[1:]] = False
    #th = np.flip(th)

    occurrences_of_true = np.where(th == True)
    
    for i in range(0,len(occurrences_of_true)):
        if(occurrences_of_true[i] == True):
            rogo_cat[i] = -1
    
    X = np.row_stack([rogo2[start:end], int_rogo2[start:end], rogo_cat])
    if (isinstance(X, list) == True):
        print('Is list')
        X = np.concatenate(X, axis=1)
    else:
        print('IS not list')
    # X = np.concatenate(X,axis=0)
    X = X.reshape((-1, 3))
    
    X = X.reshape((-1, 3))
    y = np.array(sensitivities2)
    '''
    for idx in I:
        y[idx] = y
        x[idx,1] =
    '''
    
    y = y[I]
    X[:, 0:1] = X[I, 0]
    X[:, 1:2] = X[I, 1]
    X[:, 2:3] = X[I, 2]
    # y = y.reshape((-1,1))
    # y = np.concatenate(y, axis=0)
    y = np.reshape(y, (-1, 1))
    X = np.reshape(X, (-1, 3))
    regr = linear_model.LinearRegression()
    print('X type:', type(X))
    print('y type:', type(y))
    '''
    Degree of Polynomial regression
    '''
    poly3 = PolynomialFeatures(degree=15)
    Xt3 = poly3.fit_transform(X)
    regr.fit(Xt3, y)
    Coefs = regr.coef_
    
    
    print('Slope:', Coefs)
    intercept = regr.intercept_
    print('Intercept:', intercept)
    y_hat = regr.predict(Xt3)
    r2 = regr.score(Xt3, y)
    print(r2)
    results = evaluate_model(Xt3, y, regr)
    print('Mean Linear MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))

    plt.figure()
    fig, ax = plt.subplots()
    plt.title('Regression Analysis')
    X_rogo = np.asarray(rogo_curr2[start:end])
    X_rogo = X_rogo[I]
    X_rogo = X_rogo.ravel()
    Xrogo_size = np.size(X_rogo)
    print('Rogo size: ', Xrogo_size)

    print('y size: ', y.size)
    y_raveled = y.ravel()
    plt.scatter(X_rogo, y_raveled)  # ,'r')
    y_hat_raveled = y_hat.ravel()
    plt.plot(X_rogo, y_hat_raveled, 'b')
    ax.margins(x=0,y=0)
    
    
    
    print('Done plotting regression')
    Huber = HuberRegressor()
    # xaxis = np.arange(X.min(), X.max(), 0.01)
    Huber.fit(Xt3, y.ravel())
    print('Done fitting Huber')
    CoefsHub = Huber.coef_
    yH = Huber.predict(Xt3)
    print('Done Predicting Huber')
    results = evaluate_model(Xt3, y, Huber)
    print('Mean Huber MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))
    # plot the line of best fit

    plt.figure()
    plt.title('Huber Analysis')
    plt.plot(X_rogo, yH, 'b')
    plt.scatter(X_rogo, y)  # ,'r')
    # plt.legend(['Estimate Huber', 'Measured'])

    print('Calculated Sensitivity = %0.12f V*s/A' % sensitivity2)
    sense_array2.append(sensitivity2)


    Ridger = Ridge()
    Ridger.fit(Xt3,y.ravel())
    
    CoefsRidg = Ridger.coef_
    yR = Ridger.predict(Xt3)
    
    results = evaluate_model(Xt3, y, Ridger)
    print('Mean Ridge MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))
    # plot the line of best fit
    
    plt.figure()
    plt.title('Ridge Analysis')
    plt.plot(X_rogo, yR, 'b')
    plt.scatter(X_rogo, y)  # ,'r')
    '''
    sensitivities3 = int_peaks3
    for x in range(0, len(int_peaks3)):
        sensitivities3[x] = (int_peaks3[x]/pear_peaks3[x])/(0.1)  # V*s/A
    sensitivity3 = np.mean(sensitivities3)
    print('Calculated Sensitivity = %0.13f V*s/A' % sensitivity)
    sense_array3.append(sensitivity3)
    '''
    '''
    sensitivities4 = int_peaks4
    for x in range(0,len(int_peaks4)):
        sensitivities4[x] = (int_peaks4[x]/pear_peaks4[x])/(0.1)              #V*s/A 
    sensitivity4 = np.mean(sensitivities4)
    
    print('Calculated Sensitivity = %0.12f V*s/A' % sensitivity)
    sense_array4.append(sensitivity4)
    
    '''
    '''
    sense_array2.append(sensitivity2)
    sense_array3.append(sensitivity3)
    sense_array4.append(sensitivity4)
    '''

    '''
    
    #---------------Plotting---------------------
    '''
    plt.figure()
    plt.title(file)
    plt.grid(True)
    #plt.xlim(0, 20)
    # time = [x*1e6 for x in time]
    plt.plot(time_fix, pear_curr, time_fix, rogo_curr2)
    # , time, pass_curr)
    plt.xlabel('Time(us)')
    plt.ylabel('Current (A))')
    plt.legend(['Pearson = %0.2f' % (max(pear_curr)),
                'Mathematical Rogowski = %0.2f' % max(np.abs(rogo_curr2))])
    # , 'Passive Rogowski = %0.2f' % pass_peak])

    #plt.figure(file)
    plt.grid(True)
    #plt.xlim(0, 20)
    # time = [x*1e6 for x in time]
    plt.plot(time_fix, rogo_curr2)
    # , time, rogo_curr3, time, rogo_curr4)
    plt.xlabel('Time(us)')
    plt.ylabel('Current (A))')

'''
    
#----------------Difference Calculations-------------------
'''
'''
pear_diff = (max(pear_array) - min(pear_array)) * \
    100/(sum(pear_array)/len(pear_array))
'''
# int_diff = (max(int_array) - min(int_array))*100/(sum(int_array)/len(int_array))
# pass_diff = (max(pass_array) - min(pass_array))*100/(sum(pass_array)/len(pass_array))
# sense_diff = ((max(sense_array) - min(sense_array))*100)/(sum(sense_array)/len(sense_array))
# print(pear_diff, int_diff, pass_diff, sense_diff)
# print(pear_diff, sense_diff)

average_sense2 = sum(sense_array2) / len(sense_array2)
print('Average Sensitivity2', average_sense2)
'''
average_sense3 = sum(sense_array3)/len(sense_array3)
print('Average Sensitivity3', average_sense3)
'''
'''
average_sense4 = sum(sense_array4)/len(sense_array4)
print('Average Sensitivity4', average_sense4)
'''

sense_diff2 = ((max(sense_array2) - min(sense_array2)) * 100) / (sum(sense_array2) / len(sense_array2))
'''
sense_diff3 = ((max(sense_array3) - min(sense_array3))*100)/ \
        (sum(sense_array3)/len(sense_array3))
'''
'''
sense_diff4 = ((max(sense_array4) - min(sense_array4))*100)/(sum(sense_array4)/len(sense_array4))
'''

print(sense_diff2)

'''
print(sense_diff3)
'''
'''
print(sense_diff4)
'''

average_sense2 = sum(sense_array2) / len(sense_array2)
print('Average Sensitivity', average_sense2)

'''
average_sense3 = sum(sense_array3)/len(sense_array3)
print(average_sense3)
'''
'''
average_sense4 = sum(sense_array4)/len(sense_array4)
print( average_sense4)
'''
norm_array2 = sense_array2 / average_sense2

file_num = [] * count
file_num = [string.ascii_uppercase[x] for x in range(0, count)]
plt.figure('Sensativity Differences')
plt.grid(True)
plt.xlim(0, count - 1)
plt.bar(file_num, norm_array2)
plt.plot(file_num, [1.05 for x in range(0, count)])
plt.plot(file_num, [0.95 for x in range(0, count)])
plt.xlabel('Module Name')
plt.ylabel('Normalized Sensativity Factor')
# Sensitivity', 'Pearson',
plt.legend(['>105%', '<95%', 'Normalized Sensitivities'])

print(sense_array)

plt.show()
