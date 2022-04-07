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

import datetime


# function to output current distribution and total
def curr_peaks(a, b, c, d, e, f, g, h):
    '''
    Outputs high power current values detected by Rogowski coils. Check order and number of coils for total current. 
    
    Parameters: a, b, c, d - Rogowski coil current arrays from scope

    return NONE
    '''
    # peaks of integration for current
    a_peak = max_array(a)
    b_peak = max_array(b)
    c_peak = max_array(c)
    d_peak = max_array(d)
    e_peak = max_array(e)
    f_peak = max_array(f)
    g_peak = max_array(g)
    h_peak = max_array(h)
    '''
    total_curr_3 = addition(a_peak[0], b_peak[0], c_peak[0])
    total_curr_2 = addition(d_peak[0],e_peak[0],f_peak[0])
    total_curr_1 = addition(g_peak[0],h_peak[0],0)    
    '''
    peak_array = a_peak, b_peak, c_peak, d_peak, e_peak, f_peak, g_peak, h_peak  # array to store module peaks

    # Plotting difference between each module for all files
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

    # Print the current of each Module
    print('3 West:', a_peak, 'A/s')
    print('3 Center:', b_peak, 'A/s')
    print('3 East:', c_peak, 'A/s')
    print('2 West:', d_peak, 'A/s')
    print('2 Center:', e_peak, 'A/s')
    print('2 East:', f_peak, 'A/s')
    print('1 West:', g_peak, 'A/s')
    print('1 Center:', h_peak, 'A/s')

    # print('TOTAL N3 MODULE CURRENT:', total_curr, 'A/s')

    return peak_array


# function to find max current of each signal in module
def max_array(x):
    '''
    Finds the peak of the rogowski current data
    
    Parameters: x - input current array

    returns the time and maximum peak value of the array
    '''
    peaks, _ = find_peaks(x, height=max(x))
    # plt.plot(time[peaks], x[peaks],"*")
    pfound = x[peaks[0]]

    return pfound


# function to combine a Modules total for 3 Rogowski coils
def addition(a, b, c):
    '''
    Finds the total peak current of the module of 4 high current wires  

    Parameters: a, b, c, d - peak current values    

    returns total current value
    '''

    addition = a + b + c;

    return addition


# function to add 40dB attenuation
def att(data):
    attenuation = 100  # 40dB
    tmp_x_arr = [x * attenuation for x in data]

    return tmp_x_arr


# function to filter out raw data
def filter(data):
    fs = 2.5e9  # Sampling frequency
    fc = 1e7  # cutoff frequency
    w = fc / (fs / 2)  # Normalize frequency
    b, a = signal.butter(5, w, 'low')
    filtered = signal.filtfilt(b, a, data)

    return filtered


# function to integrate the raw voltage data collected from the Rogowski coil using cumunalative sums
def integration(data, time, sense):
    dscale = 1  # e-3 #mV to V input data
    tscale = 1  # e6 #seconds to microseconds

    # scale everything
    tmp_x_arr = [x * tscale for x in time]
    tmp_y_arr = [y * dscale / sense for y in data]

    int_curr = integrate.cumulative_trapezoid(tmp_y_arr, tmp_x_arr, initial=0)  # V

    # int_curr = volt_int/sense   #A/s
    return int_curr


def checksum(csv_files, path):
    cksum_prev = "0"
    file_prev = "0"
    MD5_name = path + "\MD5_of_files_" + pd.to_datetime('today').strftime('%Y%m%d') + ".txt"
    fid = open(MD5_name, "w")
    for file in csv_files:
        head, tail = ntpath.split(file)
        cksum = hashlib.md5(file.encode('utf-8')).hexdigest()
        if (cksum is cksum_prev):
            print("File ", tail, " is a duplicate of ", file_prev, ".")
        else:
            print("File is not a copy.")
        cksum_prev = cksum
        fid.writelines([tail, "    ", cksum, "\n"])
        file_prev = tail

    fid.close()
    return 0
