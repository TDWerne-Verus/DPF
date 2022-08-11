"""
Created on Thu Mar 17 10:55:55 2022

@author: tyler.werne

Created on Wed Mar 16 21:29:27 2022

@author: tyler.werne
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

# plot settings
plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'lines.linewidth': 5})

# Global variables
sensitivity = 1  # 67.6301041736064e-9#0.217*1e-9; #V/(A/s)
sense_array = [];
pear_array = [];
int_array = [];
pass_array = [];
x_peaks_array = []
senses_array = []
rogo_array = []

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
order = 5

# import csv files from folder
path = os.getcwd();  # currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"));  # files in directory

# Checksum function
checksum(csv_files, path)
# End of checksum function

# loop over the list of csv files
pass_rogo = []
rogo = []
pear_curr = []
plt.close('all')

# Variables for each file
time = [];
pear = [];  # pearson voltage
pear_curr = [];  # pearson current
rogo = [];  # raw rogowski voltage
rogo_vol = [];  # integrated raw rogowski voltage
pass_rogo = [];  # passivly integrated rogowski current
sensitivities = [];

for file in csv_files:

    pass_curr = [];  # current of passive rogowski

    rogo_curr = [];  # rogowski current
    Ch1 = 'Ch1'
    Ch2 = 'Ch2'
    # read the csv file
    data = pd.read_csv(file, header=6)
    print(file)
    if (file.find(Ch1) > -1):
        count = count + 1
        print('Channel 1')
        data.columns = ['1', '2', '3', 'TIME', 'CH1']
        df = pd.DataFrame(data, columns=['TIME', 'CH1'])
        time = df['TIME']  # seconds
        Fs = 1000e6
        # time_fix = np.linspace(time[0], time[time.size-1],num=
        #                       int(Fs*(time[time.size-1]-time[0]))+1)

        # --------Pearson Processing----------
        pear = df['CH1'];  # attenuated voltage
        pear_att = att(pear, att_fact_pear)  # voltage
        for x in range(len(pear_att)):  # amperage
            pear_curr.append(pear_att[
                                 x] * 2 / pear_sense)  # can increase speed by pre-allocating pear_curr, multiplied by 2 due to Pearson impedances
        pear_peak = find_peak(pear_curr)
        print('Pearson current = %0.2f A' % pear_peak)
        pear_array.append(pear_peak)

    elif (file.find(Ch2) > -1):
        data.columns = ['1', '2', '3', 'TIME', 'CH2']
        print('Channel 2')
        df = pd.DataFrame(data, columns=['TIME', 'CH2'])
        time = df['TIME']  # seconds
        Fs = 1000e6
        # time_fix = np.linspace(time[0], time[time.size-1],num=
        #                       int(Fs*(time[time.size-1]-time[0]))+1)

        # --------Rogowski Voltage Processing----------
        rogo = df['CH2'];  # K*dI/dt
        int_rogo = integration(rogo, 1, time, 1)  # K*I
        # int_rogo = MAF(int_rogo, order)
        for x in range(len(int_rogo)):
            rogo_curr.append(int_rogo[x] / 4)  # divided by 4 loops
            # slow we can make faster by not dynamically allocating
        int_peak = find_peak(rogo_curr)
        # matches = (x for x in rogo_curr if x > 0.95*max(rogo_curr))
        # peak of integrated current
        # x_peaks = [x_peaks for x_peaks,x in enumerate(rogo_curr) if x > 0.95*max(rogo_curr)]
        n = 0
        '''
        for x in rogo_curr:
            if (x > 0.95*max(rogo_curr)):
                start = n
                break
            n += 1
        rogo_curr.reverse()
        n = 0
        for x in rogo_curr:
            if (x > 0.95*max(rogo_curr)):
                end_rev = n
                break
            n += 1 
        rogo_curr.reverse()
        end = len(rogo_curr) - end_rev
        '''

        PFa = 0.0001
        refLength = 500
        guardLength = 150000

        CFARThreshold = np.abs(CFAR_SS(rogo, PFa, refLength, guardLength))

        for t in range(0, len(CFARThreshold)):
            if ((rogo[t] > CFARThreshold[t]) or (rogo[t] < -1 * CFARThreshold[t])):
                start = t
                break;

        # start = start[0]
        th = np.abs(rogo_curr) > (0.60 * max(np.abs(rogo_curr)))
        th[1:][th[:-1] & th[1:]] = False
        th = np.flip(th)

        occurrences_of_true = np.where(th == True)

        if (isinstance(occurrences_of_true, int)):
            end_rev = occurrences_of_true
        else:
            end_rev = occurrences_of_true[0][0]
            if (end_rev == 0):
                end_rev = occurrences_of_true[0][1]
        end = len(rogo_curr) - end_rev
        # end = end_rev
        # end = end[0]

        # end = end[0]
        '''
        #x_peaks = [enumerate(rogo_curr)]
        y_peaks = rogo_curr[x_peaks[0]:x_peaks[-1]]
        start = x_peaks[0]
        end = x_peaks[-1]
        '''
        x_peaks = [x for x in range(start, end)]
        int_peaks = rogo_curr[start:end]
        pear_peaks = pear_curr[start:end]
        print('Integrated Rogwoski sensitivity current = %0.10f V*s' %
              np.mean(int_peak))
        int_array.append(int_peak)

        # ---------Sensitivity Calcualtions-----------
        sensitivity = (int_peak / pear_peak) / (0.1)  # V/A
        print('Calculated SP Sensitivity = %0.12f V*s/A' % sensitivity)
        sensitivities = int_peaks
        n = 0;
        m = 0
        n_peaks = x_peaks
        dels = []
        for x in range(0, len(int_peaks), 1):
            if (pear_peaks[x] != 0):
                sensitivities[n] = (int_peaks[x] / pear_peaks[x]) / (0.1)  # V*s/A
                n += 1
                m += 1
            else:
                # do nothing
                del n_peaks[m]
                print('Found Divide by zero at ', x)
        del sensitivities[n:-1]
        sensitivity = np.mean(sensitivities)

        plt.figure('Sensativity Over All Data')
        plt.grid(True)
        plt.xlim(x_peaks[0], x_peaks[-1])
        plt.bar(n_peaks, sensitivities)
        plt.xlabel('Time [s]')
        plt.ylabel('Sensitivity')

        print('Calculated Sensitivity = %0.12f V*s/A' % sensitivity)
        sense_array.append(sensitivity)
        x_peaks_array.append(x_peaks)
        senses_array.append(sensitivities)
        rogo_array.append(rogo[start:end] / 4)
    '''
    Eventually, we want to not rely on having both channels in the same file 
    folder
    if():
    '''
    # setting variables to channels and scale to base units

    # ---------------Plotting---------------------
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
# ----------------Best Fit Function------------------------------


# ----------------Difference Calculations-------------------
pear_diff = (max(pear_array) - min(pear_array)) * 100 / (sum(pear_array) /
                                                         len(pear_array))
int_diff = (max(int_array) - min(int_array)) * 100 / (sum(int_array) /
                                                      len(int_array))
'''
pass_diff = (max(pass_array) - min(pass_array))*100/(sum(pass_array)/
                                                     len(pass_array))
'''
sense_diff = ((max(sense_array) - min(sense_array)) * 100) / (sum(sense_array) /
                                                              len(sense_array))
print(pear_diff, int_diff, sense_diff)  # pass_diff

average_sense = sum(sense_array) / len(sense_array)
print('Average Sensitivity', average_sense)

norm_array = (sense_array / average_sense)

'''

'''

'''
'''

file_num = [] * count
file_num = [string.ascii_uppercase[x] for x in range(0, count)]
plt.figure('Sensativity Differences')
plt.grid(True)
plt.xlim(0, count - 1)
plt.bar(file_num, norm_array)
plt.plot(file_num, [1.05 for x in range(0, count)])
plt.plot(file_num, [0.95 for x in range(0, count)])
plt.xlabel('Module Name')
plt.ylabel('Normalized Sensativity Factor')
plt.legend(['>105%', '<95%', 'Normalized Sensitivities'])  # Sensitivity', 'Pearson',

print(sense_array)

plt.show()
