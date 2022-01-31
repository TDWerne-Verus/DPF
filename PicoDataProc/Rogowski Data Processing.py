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
from pandas import DataFrame
import os 
import glob
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
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
from RogowskiDataProcessingFunctions import *




#function to output current distribution and total
def curr_peaks(a, b, c, d, e, f):
    '''
    Outputs high power current values detected by Rogowski coils
    
    Parameters: a, b, c, d - Rogowski coil current arrays from scope


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
    peaks = curr_peaks(N3E_int, S2C_int, N3C_int, N2C_int, N1C_int, N3W_int, time)

    
    (Adding new channels for full analysis)
    #plot column arrays
    
    #Plot filtered raw data
    (Integration)
    #fig, ax1 = plt.subplots()
    #plt.grid(True)
    #plt.xlim(0,2e-5)
    #plt.plot(time,N3E_filt,time,S2C_filt,time,N3C_filt, time, N2C_filt, time, N1C_filt, time,N3W_filt)
    #plt.xlabel('Time(s)')
    #plt.ylabel('Current (A))')
    #plt.legend(['N3E','S2C','N3C','N2C','N1C','N3W'])

    #Plot integrated Rogowski coil as current
    plt.grid(True)
    plt.xlim(0,2e-5)
    plt.plot(time,N3E_int,time,S2C_int,time,N3C_int, time, N2C_int, time, N1C_int, time, N3W_int)
    plt.xlabel('Time(s)')
    plt.ylabel('Current (A)')
    plt.legend(['N3E','S2C','N3C','N2C','N1C','N3W'])
    plt.figure()
    #plt.figure
    #plt.plot([1,2,3,4,5,6],peaks)
    #plt.xlabel('Peaks of Channels')
    #plt.ylabel('Current (A)')
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
    plt.legend(['East Current','Center Current','West Current'])
    '''
    
    
    #Write data to Excel
    df = DataFrame({'N3 East': peaks[0]})
    print(df)
    df.to_excel('test.xlsx',sheet_name = 'sheet1', index = False)




plt.show()


  

    

 