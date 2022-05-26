


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:11:03 2022

@author: tyler.werne
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

import os 
import glob
import matplotlib.pyplot as plt
from matplotlib.axis import Axis


from scipy.signal import find_peaks
from scipy.integrate import quad
import hashlib
import ntpath
import datetime

#Function for integration
#def integrate(x):
   # return [np.sin(x), np.exp(x)]

#function to output current distribution and total
def distribution(a, b, c, sensitivity):
from matplotlib.figure import Figure
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
    #peaks of voltages on scope data
    a_peak = max_array(a)
    b_peak = max_array(b)
    c_peak = max_array(c)
    #d_peak = max_array(d)

    #convert voltages to currents using sensativity 
    a_current = a_peak/(sensitivity*1000)
    b_current = b_peak/(sensitivity*1000)
    c_current = c_peak/(sensitivity*1000)
    #d_current = (d_peak *1000)/sensativity

    total_curr = addition(a_current,b_current,c_current)
    total_volt = addition(a_peak, b_peak, c_peak)

    print('CHANNEL A - East side:', a_peak, 'mV = ',a_current, 'A/ns')
    print('CHANNEL E - Center:', b_peak, 'mV = ', b_current, 'A/ns')
    print('CHANNEL H - West Side:', c_peak, 'mV = ', c_current, 'A/ns')
    print('TOTAL MODULE SCOPE VOLTAGE:', total_volt, 'mV = ', total_curr, 'A/ns')


    #peaks of integration for current
    a_peak = max_array(a)
    b_peak = max_array(b)
    c_peak = max_array(c)
    d_peak = max_array(d)
    e_peak = max_array(e)
    f_peak = max_array(f)

    total_curr = addition(a_peak,c_peak,f_peak)
    
    
    #df = pd.read_excel()
    peak_array = [a_peak,b_peak,c_peak,d_peak,e_peak,f_peak]
    #peak_array.to_excel('Numerically_Integrated_Rogowski_Currents.xlsx')

    print('N3 East:', a_peak, 'A/s')
    print('N3 West:', b_peak, 'A/s')
    print('N1 Center:', c_peak, 'A/s')
    print('N2 Center:', d_peak, 'A/s')
    print('N3 Center:', e_peak, 'A/s')
    print('S2 Center:', f_peak, 'A/s')

    print('TOTAL N3 MODULE CURRENT:', total_curr, 'A/s')
    
    return peak_array

#function to find max current of each signal in module
def max_array(x):
    '''
    Finds the peak of the rogowski current data
    
    Parameters: x - input current array

    returns the time and maximum peak value of the array
    '''
    
    peaks, _ = find_peaks(x, height = max(x))
    plt.plot(time[peaks], x[peaks],"x",color="gray")
    
    return x[peaks[0]]

#function to combine the total of all of the currents in a module
    peaks, _ = find_peaks(x, height = max(x))
    #plt.plot(time[peaks], x[peaks],"*")
    pfound = x[peaks]
    
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


#function to integrate the raw voltage data collected from the Rogowski coil
def integration(data):
    '''
    
    '''
    for x in range(0,len(data)):
        print(x)
        int_curr = (quad(data[x], x-1, x+1))

    return int_curr

def checksum(csv_files, path):
    cksum_prev = "0"
    file_prev = "0"
    MD5_name = path + "\MD5_of_files_" + pd.to_datetime('today').strftime('%Y%m%d') +".txt"
    fid = open(MD5_name,"w")
    for file in csv_files:
        head, tail = ntpath.split(file)
        cksum = hashlib.md5(file.encode('utf-8')).hexdigest()
        if(cksum is cksum_prev):
            print("File ",tail," is a duplicate of ",file_prev,".")
        else:
            print("File is not a copy.")
        cksum_prev = cksum
        fid.writelines([tail,"    ", cksum,"\n"])
        file_prev = tail
        
    fid.close()
    return 0


#function to add 40dB attenuation
def att(data):
    attenuation = 100 #40dB
    tmp_x_arr = [x*attenuation for x in data]
    
    return tmp_x_arr
    
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
