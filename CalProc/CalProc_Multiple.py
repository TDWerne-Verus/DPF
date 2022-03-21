# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:29:27 2022

@author: tyler.werne
"""

"""
@Tiffany Berntsen - for Verus Research Rogowski Coil calibration for the data
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
import scipy.special as special
from scipy.integrate import simps
from numpy import trapz
import datetime
from ProcessingFunctions import *
import os
from matplotlib import pyplot as plt 
import string


#plot settings
plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'lines.linewidth': 5})

#Global variables
sensitivity = 1#67.6301041736064e-9#0.217*1e-9; #V/(A/s)
sense_array = [];
pear_array = [];
int_array = [];
pass_array = [];

count = 0
pear_sense = 0.1           #pearson sensativity is 0.1 V/A
att_fact_pear = 40.0         #attinuation factor of pearson
load = 2                   #dependant on what the pearson is expecting 
rise_time = 5.6*1e-6         #typical rise time of signal
R = 50                     #resistance in calibration system 

#Oscope scaling
pscale = 0.5
rvscale = 2
rascale = 0.05

#import csv files from folder
path = os.getcwd();                                      #currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"));      #files in directory

# Checksum function
checksum(csv_files, path)
# End of checksum function

#loop over the list of csv files
pass_rogo = []
rogo= []
pear_curr = []
plt.close('all')

for file in csv_files:
    
    #Variables for each file
    time = [];
    pear = [];      #pearson voltage
    pear_curr = []; #pearson current
    rogo = [];      #raw rogowski voltage
    rogo_vol = []   #integrated raw rogowski voltage
    rogo_curr = []; #rogowski current
    pass_rogo = []; #passivly integrated rogowski current
    pass_curr = []; #current of passive rogowski
    
    #read the csv file
    data = pd.read_csv(file, header= 14) 
    count = count+1
    print(file)
    
    #setting variables to channels and scale to base units
    df = pd.DataFrame(data, columns= ['TIME','CH1','CH2','CH3'])
    time = df['TIME']                                   #seconds
    Fs = 1000e6
    time_fix = np.linspace(time[0], time[time.size-1],num=
                           int(Fs*(time[time.size-1]-time[0]))+1)
    
    #--------Pearson Processing----------
    pear = df['CH1'];                                   #attenuated voltage
    pear_att = att(pear, att_fact_pear)                 #voltage
    for x in range(len(pear_att)):                      #amperage
        pear_curr.append(pear_att[x]*2/pear_sense)      # can increase speed by pre-allocating pear_curr, multiplied by 2 due to Pearson impedances
    pear_peak = find_peak(pear_curr)
    print('Pearson current = %0.2f A' % pear_peak)
    pear_array.append(pear_peak)
    
    #--------Rogowski Voltage Processing----------
    rogo = df['CH2'];                                    #K*dI/dt
    int_rogo = integration(rogo, 1, time_fix, 1)         #K*I
    for x in range(len(int_rogo)):
        rogo_curr.append(int_rogo[x]/4)                  #divided by 4 loops
    int_peak = find_peak(rogo_curr)                      #peak of integrated current
    print('Integrated Rogwoski sensitivity current = %0.10f V*s' % int_peak)
    int_array.append(int_peak)
    
    #---------Rogowski Current Processing----------
    pass_rogo = df['CH3'];                              #amperage
    for x in range(len(pass_rogo)):
        pass_curr.append(pass_rogo[x]/4)                #divided by 4 loops
    pass_peak = find_peak(pass_curr)                    #peak of passive current
    print('Passive Rogwoski sensitivity current = %0.10f V*s' % pass_peak)
    pass_array.append(pass_peak)
    
   #---------Sensitivity Calcualtions-----------
   # sensitivity = (int_peak/pass_peak)/(0.1)       #V/A
    sensitivity = (int_peak/pear_peak)              #V*s/A 
    print('Calculated Sensitivity = %0.12f V*s/A' % sensitivity)
    sense_array.append(sensitivity)

    #---------------Plotting---------------------
    '''
    plt.figure(file)
    plt.grid(True)
    plt.xlim(0,20)
    time = [x*1e6 for x in time]
    plt.plot(time, pear_curr, time, rogo_curr, time, pass_cuSrr)
    plt.xlabel('Time(us)')
    plt.ylabel('Current (A))')
    plt.legend(['Pearson = %0.2f' % pear_peak, 'Mathematical Rogowski = %0.2f' % int_peak, 'Passive Rogowski = %0.2f' % pass_peak])
   '''
    
#----------------Difference Calculations-------------------
pear_diff = (max(pear_array) - min(pear_array))*100/(sum(pear_array)/
                                                     len(pear_array))
int_diff = (max(int_array) - min(int_array))*100/(sum(int_array)/
                                                  len(int_array))
pass_diff = (max(pass_array) - min(pass_array))*100/(sum(pass_array)/
                                                     len(pass_array))
sense_diff = ((max(sense_array) - min(sense_array))*100)/(sum(sense_array)/
                                                          len(sense_array))
print(pear_diff, int_diff, pass_diff, sense_diff)

average_sense = sum(sense_array)/len(sense_array)
print('Average Sensitivity', average_sense)

norm_array = (sense_array/average_sense)

'''

'''
N_array = sense_array[0:3]
O_array = sense_array[3:6]
Q_array = sense_array[6:9]

'''
'''

file_num = []*count
file_num = [string.ascii_uppercase[x] for x in range(0, count)]
plt.figure('Sensativity Differences')
plt.grid(True)
plt.xlim(0, count-1)
plt.bar(file_num, norm_array)
plt.plot(file_num, [1.05 for x in range(0,count)])
plt.plot(file_num, [0.95 for x in range(0,count)])
plt.xlabel('Module Name')
plt.ylabel('Normalized Sensativity Factor')
plt.legend(['>105%', '<95%','Normalized Sensitivities']) #Sensitivity', 'Pearson',


plt.figure('Sensativity Differences N')
plt.grid(True)
plt.xlim(0, count-1)
plt.bar([1,2,3], N_array/np.mean(N_array))
plt.plot([1,2,3], [1.05 for x in range(0,3)])
plt.plot([1,2,3], [0.95 for x in range(0,3)])
plt.xlabel('Module Name')
plt.ylabel('Normalized Sensativity Factor')
plt.legend(['>105%', '<95%','Normalized Sensitivities']) #Sensitivity', 'Pearson',

plt.figure('Sensativity Differences O')
plt.grid(True)
plt.xlim(0, count-1)
plt.bar([1,2,3], O_array/np.mean(O_array))
plt.plot([1,2,3], [1.05 for x in range(0,3)])
plt.plot([1,2,3], [0.95 for x in range(0,3)])
plt.xlabel('Module Name')
plt.ylabel('Normalized Sensativity Factor')
plt.legend(['>105%', '<95%','Normalized Sensitivities']) #Sensitivity', 'Pearson',

print(sense_array)

plt.show()
  


