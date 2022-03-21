# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:29:53 2022

@author: tyler.werne
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:11:03 2022

@author: Tiffany Berntsen and Tyler Werne
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
import hashlib
import ntpath

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

def integration(data, dscale, time, tscale):
    '''

    Parameters
    ----------
    data : array of floats
        array of to be integrated
    dscale : integer
        scaling factor of data to convert to voltage
    time : array of floats
        array of time data
    tscale : integer
        scaling factor of time to convert into seconds

    Returns
    -------
    int_curr : array of floats
       trapazoidal integration array

    '''

    #scale everything
    tmp_x_arr = [x*tscale for x in time]
    tmp_y_arr = [y*dscale for y in data]

    int_arr = integrate.cumulative_trapezoid(tmp_y_arr, tmp_x_arr, initial = 0)

    return int_arr



def att(data, att_fact):
    '''
    
    Parameters
    ----------
    data : float array
        array of float that describe voltage
    att_fact : integer
        attinuation number in dB

    Returns
    -------
    tmp_x_arr : float array
        array of floats that is the attenuated voltage value

    '''
    
    attenuation = 10**(att_fact/20)
    tmp_x_arr = [x*attenuation for x in data]
    
    return tmp_x_arr

def find_peak(data_array):
    '''
    Parameters
    ----------
    data_array : array of floats
        data array where the first peak of the array needs to be found

    Returns
    -------
    pfound : float
        first peak of array found

    '''
    peaks, _ = find_peaks(data_array, height = max(data_array))
   
    pfound = max(data_array)
    
    return pfound


def MAF(data_array ,order):
    '''
    Moving-Average Filter: Low pass filter
    
    Parameters
    ----------
    data_array : array of floats
        data array where the first peak of the array needs to be found
        
    order : int
        integer specifying order of the Moving-Average Filter

    Returns
    -------
    data_MAF : array of floats
        data with MAF applied

    '''
    data_MAF = data_array
    for x in range(order,len(data_array)):
        data_MAF[x] = np.sum(data_array[x+1-order:x+1])/order
        
    return data_MAF




#function to output current distribution and total
def curr_peaks(a, b, c, d, e):
    '''
    Outputs high power current values detected by Rogowski coils. Check order and number of coils for total current. 
    
    Parameters: a, b, c, d - Rogowski coil current arrays from scope

    return NONE
    '''
    #peaks of integration for current
    a_peak = max_array(a)
    b_peak = max_array(b)
    c_peak = max_array(c)
    d_peak = max_array(d)
    e_peak = max_array(e)
    #f_peak = max_array(f)


    #total_curr = addition(c_peak[0],d_peak[0],e_peak[0])     #total of N3 Capacitor bank
    
    peak_array = a_peak,b_peak,c_peak,d_peak,e_peak #array to store module peaks
    
    #Plotting difference between each module for all files
    '''plt.rcParams.update({'font.size': 35})
    plt.rcParams.update({'lines.linewidth': 14})
    plt.figure('Scaled Current between Modules')
    plt.grid(True)
    plt.xlabel('Modules')
    x=[1,2,3,4,5]
    labels = ['N3E', 'N3W', 'N1C', 'N2C', 'N3C']
    plt.xticks(x, labels, rotation='vertical')
    plt.ylabel('Current (A)')
    plt.plot(x,[a_peak,b_peak,c_peak,d_peak,e_peak])
    #plt.legend(['Shot 11']) # 45kV on 1/26/22
    #plt.legend(['Shot 1','Shot 2','Shot 4','Shot 8','Shot 10','Shot 12']) #30kV on 1/26/22
    #plt.legend(['Shot 1','Shot 3','Shot 7','Shot 9','Shot 11','Shot 13']) #30kV on 1/27/22
    plt.legend(['Shot 4','Shot 6','Shot 10']) #45kV on 1/27/22'''

    #Print the current of each Module
    print('N3 East:', a_peak, 'A/s')
    print('N3 West:', b_peak, 'A/s')
    print('N1 Center:', c_peak, 'A/s')
    print('N2 Center:', d_peak, 'A/s')
    print('N3 Center:', e_peak, 'A/s')
    #print('S2 Center:', f_peak, 'A/s')

    #print('TOTAL N3 MODULE CURRENT:', total_curr, 'A/s')
    
    return peak_array

#function to find max current of each signal in module
def max_array(x):
    '''
    Finds the peak of the rogowski current data
    
    Parameters: x - input current array

    returns the time and maximum peak value of the array
    '''
    peaks, _ = find_peaks(x, height = max(x))
    #plt.plot(time[peaks], x[peaks],"*")
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



