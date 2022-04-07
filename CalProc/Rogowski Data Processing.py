"""
@Tiffany Berntsen - for Verus Research Rogowski Coil Data processing code
In this version, the Rogowski coil data is plotted and the peaks are considered the current. 
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
from RogowskiDataProcessingFunctions import *
import os
from matplotlib import pyplot as plt

# plot settings
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'lines.linewidth': 5})

# variables
sensitivity = 67.6301041736064e-9  # 0.217*1e-9; #V/(A/s)
pico_0_sense = [67.6301041736064e-9, 67.6301041736064e-9, 67.6301041736064e-9, 67.6301041736064e-9, 67.6301041736064e-9,
                67.6301041736064e-9, 67.6301041736064e-9, 67.6301041736064e-9]
pico_1_sense = []
time = [];
raw = [];
intRog = [];

peak_N3E = []
peak_S2C = []
peak_N3C = []
peak_N2C = []
peak_N1C = []
peak_N3W = []

count = 0

# import csv files from folder
path = os.getcwd();  # currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"));  # files in directory

# Checksum function
checksum(csv_files, path)
# End of checksum function

# loop over the list of csv files
for file in csv_files:
    # read the csv file
    data = pd.read_csv(file)
    count = count + 1
    print(file)

    # setting variables to current channels
    df = pd.DataFrame(data, columns=['index', 'time', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

    time = df['time']

    # pico scope 0
    N3W = df['A']  # I
    N3C = df['B']  # G
    N3E = df['C']  # A
    N2W = df['D']  # C
    N2C = df['E']  # D
    N2E = df['F']  # E
    N1W = -df['G']  # P
    N1C = df['F']  # F

    # Adding 40dB attenuation to files, all signals should be 50 Ohm terminated
    # pico scope 0
    N3W = att(N3W)
    N3C = att(N3C)
    N3E = att(N3E)
    N2W = att(N2W)
    N2C = att(N2C)
    N2E = att(N2E)
    N1W = att(N1W)
    N1C = att(N1C)
    # plt.plot(time, N3W, time, N3C, time, N3E, time, N2W, time, N2C, time, N2E, time, N1W, time, N1C)
    # plt.legend(['N3W','N3C','N3E','N2W','N2C','N2E','N1W','N1C'])

    # Integrate filtered data to get current
    N3W_int = integration(N3W, time, pico_0_sense[0])
    N3C_int = integration(N3C, time, pico_0_sense[1])
    N3E_int = integration(N3E, time, pico_0_sense[2])
    N2W_int = integration(N2W, time, pico_0_sense[3])
    N2C_int = integration(N2C, time, pico_0_sense[4])
    N2E_int = integration(N2E, time, pico_0_sense[5])
    N1W_int = integration(N1W, time, pico_0_sense[6])
    N1C_int = integration(N1C, time, pico_0_sense[7])

    # Print out the max of each current plot and total
    peaks_0 = curr_peaks(N3W_int, N3C_int, N3E_int, N2W_int, N2C_int, N2E_int, N1W_int, N1C_int)

    # Creates list of currents for each module for a full day's shot series
    peak_N3W.append(peaks_0[0])
    peak_N3C.append(peaks_0[1])
    peak_N3E.append(peaks_0[2])
    peak_N2W.append(peaks_0[3])
    peak_N2C.append(peaks_0[4])
    peak_N2E.append(peaks_0[5])
    peak_N1W.append(peaks_0[6])
    peak_N1C.append(peaks_0[7])

    '''
    #pico scope 1 
    S3W = df['A'] #K
    S3C = df['B'] #H
    S3E = df['C'] #B
    S2W = df['D'] #Q
    S2C = df['E'] #N
    S2E = df['F'] #O
    S1W = df['G'] #M
    S1C = df['F'] #J
    
    #Adding 40dB attenuation to files, all signals should be 50 Ohm terminated
    #pico scope 1
    S3W = att(S3W)
    S3C = att(S3C)
    S3E = att(S3E)
    S2W = att(S2W)
    S2C = att(S2C)
    S2E = att(S2E)
    S1W = att(S1W)
    S1C = att(S1C)
    
    #Integrate filtered data to get current
    S3W_int = integration(S3W, time, pico_1_sense[0])
    S3C_int = integration(S3C, time, pico_1_sense[1])
    S3E_int = integration(S3E, time, pico_1_sense[2])
    S2W_int = integration(S2W, time, pico_1_sense[3])
    S2C_int = integration(S2C, time, pico_1_sense[4])
    S2E_int = integration(S2E, time, pico_1_sense[5])
    S1W_int = integration(S1W, time, pico_1_sense[6])
    S1C_int = integration(S1C, time, pico_1_sense[7])
    
    #Print out the max of each current plot and total
    peaks_1 = curr_peaks(S3W_int, S3C_int, S3E_int, S2W_int, S2C_int, S2E_int, S1W_int, S1C_int)
    
    #Creates list of currents for each module for a full day's shot series      
    peak_S3W.append(peaks_1[0])
    peak_S3C.append(peaks_1[1])
    peak_S3E.append(peaks_1[2])
    peak_S2W.append(peaks_1[3])
    peak_S2C.append(peaks_1[4])
    peak_S2E.append(peaks_1[5])
    peak_S1W.append(peaks_1[6])
    peak_S1C.append(peaks_1[7])
    
    '''

    '''
    #Print out the max of each voltage
    voltage = curr_peaks(N3E,N3C,N2C,N1C,N3W)
    print(voltage)
    avg = sum(voltage)/len(voltage)
    print(avg)
    
    #plt.plot(time, N3W_int)
    '''

    # Calculates the differential between bank N3 modules
    '''dif12 = (peaks[4]-peaks[3])/peaks[3]
    dif23 = (peaks[3]-peaks[2])/peaks[2]
    dif13 = (peaks[2]-peaks[4])/peaks[4]

 
    print('N1C to N2C %f' %dif12)
    print('N2C to N3C %f' %dif23)
    print('N1C to N3C %f' %dif13)
    '''

    # Plot raw data
    plt.figure()
    plt.grid(True)
    plt.xlim(0, 2e-5)
    plt.plot(time, N3E_filt)  # time,S2C,time,N3C, time, N2C, time, N1C, time,N3W)
    plt.xlabel('Time(s)')
    plt.ylabel('Current (A))')
    # plt.legend(['N3E','S2C','N3C','N2C','N1C','N3W'])

    # Plot filtered raw data
    # plt.figure()
    # plt.grid(True)
    # plt.xlim(0,2e-5)
    # plt.plot(time,N3E_filt,time,S2C_filt,time,N3C_filt, time, N2C_filt, time, N1C_filt, time,N3W_filt)
    # plt.xlabel('Time(s)')
    # plt.ylabel('Current (A))')
    # plt.legend(['N3E','S2C','N3C','N2C','N1C','N3W'])

    # Plot integrated Rogowski coil as current
    plt.figure(file)
    plt.grid(True)
    plt.xlim(0, 2e-5)
    plt.plot(time, N3E_int, time, N3W_int, time, N1C_int, time, N2C_int, time, N3C_int)
    plt.xlabel('Time(s)')
    plt.ylabel('Current (A)')
    plt.legend(['N3E Current, Peak = %.2f' % peaks[0], 'N3W Current, Peak = %.2f' % peaks[1],
                'N1C Current, Peak = %.2f' % peaks[2], 'N2C Current, Peak = %.2f' % peaks[3],
                'N3C Current, Peak = %.2f' % peaks[4]])

    ''' plt.figure('Scaled Current between Modules')
    plt.grid(True)
    plt.xlim(0,2e-5)
    plt.xlabel('Shot #')
    plt.ylabel('Current (A)')
    plt.plot()'''
    # plt.figure
    # plt.plot([1,2,3,4,5,6],peaks)
    # plt.xlabel('Peaks of Channels')
    # plt.ylabel('Current (A)')
    # ax1.set_xlabel('Time(s)')
    # ax1.set_ylabel('Current(mV)')
    # plt.plot(time, int_A)
    # plt.legend(['Passive Integrator'])

    # ax2 = ax1.twinx()    #instantiate a second azes that shares the same x-axis
    # ax2.set_ylabel('Current(A)')

    # plt.plot(time, east, time, east_filt)
    # plt.xlabel('Time(s)')
    # plt.ylabel('Voltage(mV)')

'''
#Averaging the N3E module currents
N3Eavg = [sum(peak_N3E)/count]
N3Wavg = [sum(peak_N3W)/count]
N1Cavg = [sum(peak_N1C)/count]
N2Cavg = [sum(peak_N2C)/count]
N3Cavg = [sum(peak_N3C)/count]
print(N3Eavg)
#S2Cavg = [sum(peak_N3E)/count]

#Plot averages for a shot number
shot = 0 #shot number of the day 

plt.figure('30kV Averages for each Module')
mod_avgs = [N3Eavg,N3Wavg,N1Cavg,N2Cavg,N3Cavg]
x=[1,2,3,4,5]
labels = ['N3E', 'N3W', 'N1C', 'N2C', 'N3C']
plt.xticks(x, labels, rotation='vertical')
plt.xlim(0,6)
plt.plot(x,[N3Eavg[shot],N3Wavg[shot],N1Cavg[shot],N2Cavg[shot],N3Cavg[shot]])
plt.xlabel('Module')
plt.ylabel('Current (A)')
plt.legend(['30kV Averages for Shot series'])

-
diffs = [None]*count
for i in range(0,count):
    diffs[i] = abs((peak_N3E[i] - peak_N3E[i-1])/peak_N3E[i-1])*100
print(diffs)
# place a text box in upper left in axes coords
textstr = '\n'.join((
    r'$\=%.2f$' % (mu, ),
    r'$\mathrm{median}=%.2f$' % (median, ),
    r'$\sigma=%.2f$' % (sigma, )))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
-

#plt.figure('S2C Coil')
#plt.plot(range(0,count), peak_S2C)
#plt.xlabel('File Index')
#plt.ylabel('Current (A)')
#var = (max(peak_S2C)-min(peak_S2C))*100 /min(peak_S2C)
#plt.legend(['Percent Variation = %6.2f' %var])

#Plot module for shot series and find average
avg = [sum(peak_N3C)/count]*count 
plt.figure('N3C Module Coil for 30kV Shot Series')
plt.plot(range(0,count), peak_N3C)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
var = (max(peak_N3C)-min(peak_N3C))*100 /min(peak_N3C)
plt.legend(['Percent Variation = %6.2f' %var])

avg = [sum(peak_N2C)/count]*count 
plt.figure('N2C Module Coil for 30kV Shot Series')
plt.plot(range(0,count), peak_N2C, range(0,count), avg)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
plt.legend(['N2C Currents between Shots','Average of N2C Currents = %6.2f' %avg[0]])
#var = (max(peak_N2C)-min(peak_N2C))*100 /min(peak_N2C)
#plt.legend(['Percent Variation = %6.2f' %var])

avg = [sum(peak_N1C)/count]*count 
plt.figure('N1C Module Coil for 30kV Shot Series')
plt.plot(range(0,count), peak_N1C)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
var = (max(peak_N1C)-min(peak_N1C))*100 /min(peak_N1C)
plt.legend(['Percent Variation = %6.2f' %var])

avg = [sum(peak_N3W)/count]*count 
plt.figure('N3W Module Coil for 30kV Shot Series')
plt.plot(range(0,count), peak_N3W)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
var = (max(peak_N3W)-min(peak_N3W))*100 /min(peak_N3W)
plt.legend(['Percent Variation = %6.2f' %var])

-
plt.figure('Scaled Current Comparisons')
plt.plot(range(1,count+1), peak_N1C, range(1,count+1), peak_N2C, range(1,count+1), peak_N3C)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
varN1C = (max(peak_N1C)-min(peak_N1C))*100 /min(peak_N1C)
varN2C = (max(peak_N2C)-min(peak_N2C))*100 /min(peak_N2C)
varN3C = (max(peak_N3C)-min(peak_N3C))*100 /min(peak_N3C)
N12diff = (max(peak_N1C)-max(peak_N2C))*100/ max(peak_N2C)
N23diff = (max(peak_N2C)-max(peak_N3C))*100/ max(peak_N3C)
plt.legend(['N1C','N2C','N3C'])
-
'''
plt.show()
