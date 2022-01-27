"""
@Tiffany Berntsen - for Verus Research Rogowski Coil Data processing code
In this version, the Rogowski coil data is plotted and the peaks are considered the current. 
"""

import numpy as np
import pandas as pd
import os 
import glob
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from scipy.signal import find_peaks
from scipy import signal
from scipy import integrate
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import simps
from numpy import trapz
import datetime


#function to output current distribution and total
def distribution(a, b, c):
    '''
    Outputs high power current values detected by Rogowski coils
    
    Parameters: a, b, c, d - Rogowski coil current arrays from scope

    return NONE
    '''
    #peaks of integration for current
    a_peak = max_array(a)
    b_peak = max_array(b)
    c_peak = max_array(c)
    #d_peak = max_array(d)

    #convert from uA/s to A/s
    a_current = a_peak/(1000000)
    b_current = b_peak/(1000000)
    c_current = c_peak/(1000000)

    total_curr = addition(a_current,b_current,c_current)

    print('CHANNEL A - East side:', a_current, 'A/s')
    print('CHANNEL E - Center:', b_current, 'A/s')
    print('CHANNEL H - West Side:', c_current, 'A/s')
    print('TOTAL MODULE SCOPE VOLTAGE:', total_curr, 'A/s')

#function to find max current of each signal in module
def max_array(x):
    '''
    Finds the peak of the rogowski current data
    
    Parameters: x - input current array

    returns the time and maximum peak value of the array
    '''
    
    peaks, _ = find_peaks(x, height = max(x))
    #plt.plot(time[peaks], x[peaks],"x",color="gray")
    
    return x[peaks[0]]

#function to combine the total of all of the currents in a module
def addition(a,b,c):
    '''
    Finds the total peak current of the module of 4 high current wires  

    Parameters: a, b, c, d - peak current values    

    returns total current value
    '''

    addition = a + b + c;

    return addition

#function to filter out raw data
def filter(data):
    fs = 2.5e9  #Sampling frequency
    fc = 1e7   #cutoff frequency
    w = fc/(fs/2)   #Normalize frequency
    b,a = signal.butter(5,w,'low')
    filtered = signal.filtfilt(b,a,data)

    return filtered

#function to integrate the raw voltage data collected from the Rogowski coil using cumunalative sums
def integration(data, time, sense):
    dscale = 1e6 #I is scaled to MA, set to A
    tscale = 1e5

    tmp_x_arr = time
    tmp_y_arr = [y*dscale for y in data]

    integrated_Idot_x_arr = [x*tscale for x in tmp_x_arr]
    int_curr = integrate.cumtrapz(tmp_y_arr, integrated_Idot_x_arr, initial = 0)

    return int_curr

#import csv files from folder
path = os.getcwd();                                     #currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"));      #files in directory

#variables
sensitivity = 0.217; #V/(A/ns)
time = [];
raw = [];
intRog = [];

#loop over the list of csv files
for file in csv_files:
    #read the csv file
    data = pd.read_csv(file) 

    #setting variables to current channels
    df = pd.DataFrame(data, columns= ['index','time','A','B','C','E','H'])

    time = df['time']
    east = df['A']
    int_A = df['B'] #40dB attenuation
    dead = df['C']
    center = df['E']
    center = -center
    west = df['H']

    east_filt = filter(east)    #filter raw data
    east_curr = integration(east_filt,time,sensitivity)

    #plot column arrays
    #fig, ax1 = plt.subplots()
    plt.grid(True)
    plt.xlim(0,2e-5)
    #ax1.set_xlabel('Time(s)')
    #ax1.set_ylabel('Current(mV)')
    plt.plot(time, east_filt, time, int_A, time, east_curr)
    #plt.legend(['Current Integrated Signal'])

    #ax2 = ax1.twinx()    #instantiate a second azes that shares the same x-axis
    #ax2.set_ylabel('Current(A)')
    #plt.plot(time, east_int)
    #plt.legend(['Current Integration Calculation'])
    #plt.plot(time, east, time, east_filt)
    #plt.xlabel('Time(s)')
    #plt.ylabel('Voltage(mV)')

    #Finds distirubution of current data
    #distribution(east_curr, center, west)



plt.show()
    

  

    

 