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
def curr_peaks(a, b, c, d, e, f):
    '''
    Outputs high power current values detected by Rogowski coils
    
    Parameters: a, b, c, d - Rogowski coil current arrays from scope

    return NONE
    '''
    #peaks of integration for current
    a_peak = max_array(a)
    b_peak = max_array(b)
    c_peak = max_array(c)
    d_peak = max_array(d)
    e_peak = max_array(e)
    f_peak = max_array(f)

    total_curr = addition(a_peak,c_peak,f_peak)
    peak_array = [a_peak,b_peak,c_peak,d_peak,e_peak,f_peak]

    print('N3 East:', a_peak, 'A/s')
    print('S2 Center:', b_peak, 'A/s')
    print('N3 Center:', c_peak, 'A/s')
    print('N2 Center:', d_peak, 'A/s')
    print('N1 Center:', e_peak, 'A/s')
    print('N3 West:', f_peak, 'A/s')

    print('TOTAL N3 MODULE CURRENT:', total_curr, 'A/s')
    return peak_array

#function to find max current of each signal in module
def max_array(x):
    '''
    Finds the peak of the rogowski current data
    
    Parameters: x - input current array

    returns the time and maximum peak value of the array
    '''
    print('ITS IN HERE')
    peaks, _ = find_peaks(x, height = max(x))
    print(peaks[0])
    print(x[peaks[0]])
    plt.plot(time[peaks], x[peaks],"*",color="red")
    pfound = x[peaks[0]]
    
    return pfound

#function to combine a Modules total for 3 Rogowski coils
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
    dscale = 1e-3 #mV to V input data
    tscale = 1 #1e-5 #10 microseconds to seconds

    #scale everything
    tmp_x_arr = [x*tscale for x in time]
    tmp_y_arr = [y*dscale for y in data]

    volt_int = integrate.cumtrapz(tmp_y_arr, tmp_x_arr, initial = 0) #V

    int_curr = volt_int/sense   #A/s
    return int_curr

#import csv files from folder
path = os.getcwd();                                     #currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"));      #files in directory

#variables
sensitivity = 0.217*1e-9; #V/(A/s)
time = [];
raw = [];
intRog = [];

#loop over the list of csv files
for file in csv_files:
    #read the csv file
    data = pd.read_csv(file) 

    #setting variables to current channels
    df = pd.DataFrame(data, columns= ['index','time','A','B','C','D','E','F','G','H'])

    time = df['time']
    N3E = df['A']
    int_A = df['B'] #40dB attenuation
    dead = df['C']
    S2C = df['D']
    N3C = df['E']
    N3C = -N3C      #Inverted Rogowski Coil
    N2C = df['F']
    N1C = df['G']
    N3W = df['H']

    #filtering of raw data
    N3E_filt = filter(N3E)
    S2C_filt = filter(S2C)
    N3C_filt = filter(N3C)
    N2C_filt = filter(N2C)
    N1C_filt = filter(N1C)
    N3W_filt = filter(N3W)

    #Integrate to get current
    N3E_int = integration(N3E_filt, time, sensitivity)
    S2C_int = integration(S2C_filt, time, sensitivity)
    N3C_int = integration(N3C_filt, time, sensitivity)
    N2C_int = integration(N2C_filt, time, sensitivity)
    N1C_int = integration(N1C_filt, time, sensitivity)
    N3W_int = integration(N3W_filt, time, sensitivity)

    #Print out the max of each current plot and total
    peaks = curr_peaks(N3E_int, S2C_int, N3C_int, N2C_int, N1C_int, N3W_int)

    #plot column arrays
    #fig, ax1 = plt.subplots()
    plt.grid(True)
    plt.xlim(0,2e-5)
    plt.plot(time,N3E_filt,time,S2C_filt,time,N3C_filt, time, N2C_filt, time, N1C_filt, time,N3W_filt)
    plt.xlabel('Time(s)')
    plt.ylabel('Rogowski Voltage Input (mV)')
    plt.legend(['A','B','D','E','F','G','H'])

    plt.figure
    plt.grid(True)
    plt.xlim(0,2e-5)
    plt.plot(time,N3E_int,time,S2C_int,time,N3C_int, time, N2C_int, time, N1C_int, time,N3W_int)
    plt.xlabel('Time(s)')
    plt.ylabel('Current (A)')
    plt.legend(['A','B','D','E','F','G','H'])

    plt.figure
    plt.plot(peaks)
    plt.xlabel('Peaks of Channels')
    plt.ylabel('Current (A)')
    #ax1.set_xlabel('Time(s)')
    #ax1.set_ylabel('Current(mV)')
    #plt.plot(time, int_A)
    #plt.legend(['Passive Integrator'])

    #ax2 = ax1.twinx()    #instantiate a second azes that shares the same x-axis
    #ax2.set_ylabel('Current(A)')
    
    #plt.plot(time, east, time, east_filt)
    #plt.xlabel('Time(s)')
    #plt.ylabel('Voltage(mV)')

    #Finds distirubution of current data
    #distribution(east_curr, center, west)



plt.show()
    

  

    

 