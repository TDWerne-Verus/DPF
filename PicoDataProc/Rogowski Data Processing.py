"""
@Tiffany Berntsen - for Verus Research Rogowski Coil Data processing code
In this version, the Rogowski coil data is plotted and the peaks are considered the current. 
"""

"""
@Tyler Werne - Adding checksum functionality to check for duplicates as well as 
creating a record for data integrity.
"""
import numpy as np
import pandas as pd
import os 
import glob
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from scipy.signal import find_peaks
from scipy.integrate import quad
import hashlib
import ntpath
import datetime
from RogowskiDataProcessingFunctions import *
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

    total_curr = addition(a_peak,b_peak,c_peak)

    print('CHANNEL A - East side:', a_peak, 'A/s')
    print('CHANNEL E - Center:', b_peak, 'A/s')
    print('CHANNEL H - West Side:', c_peak, 'A/s')
    print('TOTAL MODULE SCOPE VOLTAGE:', total_curr, 'A/s')

#function to find max current of each signal in module
def max_array(x):
    '''
    Finds the peak of the rogowski current data
    
    Parameters: x - input current array

    returns the time and maximum peak value of the array
    '''
    
    peaks, _ = find_peaks(x, height = max(x))
    plt.plot(time[peaks], x[peaks],"*",color="red")
    
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

# Checksum function
checksum(csv_files, path)
# End of checksum function

#loop over the list of csv files
for file in csv_files:
    #read the csv file
    data = pd.read_csv(file) 
    head, tail = ntpath.split(file)
    print('File: ', tail)

    #setting variables to current channels
    df = pd.DataFrame(data, columns= ['index','time','A','B','C','D','E','F','G','H'])

    time = df['time']
    N3E = df['A']
    int_A = df['B'] #40dB attenuation
    dead = df['C']
    center = df['E']
    center = -center
    west = df['H']
    '''
    east_curr = integration(east)

    east_filt = filter(east)    #filter raw data
    east_curr = integration(east_filt,time,sensitivity)
    S2C = df['D']
    N3C = df['E']
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


    #east_curr = integration(east_filt,time,sensitivity)
   #distribution(east_curr, east_curr, east_curr)

    (Adding new channels for full analysis)
    #plot column arrays
    #fig, ax1 = plt.subplots()
    plt.grid(True)
    plt.xlim(0,2e-5)
    #ax1.set_xlabel('Time(s)')
    #ax1.set_ylabel('Current(mV)')
    #plt.plot(time, int_A)
    #plt.legend(['Passive Integrator'])

    #ax2 = ax1.twinx()    #instantiate a second azes that shares the same x-axis
    #ax2.set_ylabel('Current(A)')
    plt.plot(time,N3E,time,S2C,time,N3C, time, N2C, time, N1C, time,N3W)
    plt.xlabel('Time(s)')
    plt.ylabel('Current(A)')
    plt.legend(['A','B','D','E','F','G','H'])
    #plt.plot(time, east, time, east_filt)
    #plt.xlabel('Time(s)')
    #plt.ylabel('Voltage(mV)')

    #Finds distirubution of current data
    #distribution(east_curr, center, west)
    plt.legend(['East Current','Center Current','West Current'])
    '''
plt.show()


  

    

 