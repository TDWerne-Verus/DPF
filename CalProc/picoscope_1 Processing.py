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
from ProcessingFunctions import *
import os
from matplotlib import pyplot as plt 


#plot settings
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'lines.linewidth': 5})

#variables
tscale = 1
sensitivity = 67.6301041736064e-9#0.217*1e-9; #V/(A/s)
pico_1_sense = [6.28E-10,
6.36E-10,
1.11E-09,
6.72E-10,
5.52E-10,
5.91E-10,
6.66E-10,
1.10E-09]   
 
time = [];
raw = [];
intRog = [];

peak_S3W=[]
peak_S3C=[]
peak_S3E=[]
peak_S2W=[]
peak_S2C=[]
peak_S2E=[]
peak_S1W=[]
peak_S1C=[]

count = 0

#import csv files from folder
path = os.getcwd();                                      #currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"));      #files in directory

# Checksum function
checksum(csv_files, path)
# End of checksum function

#loop over the list of csv files
for file in csv_files:
    #read the csv file
    data = pd.read_csv(file) 
    count = count+1
    print(file)
    
    #setting variables to current channels
    df = pd.DataFrame(data, columns= ['index','time','A','B','C','D','E','F','G','H'])

    time = df['time']

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
    
    #Plot integrated Rogowski coil as current
    plt.figure(file)
    plt.grid(True)
    plt.xlim(0,2e-5)
    plt.plot(time, N3W_int, time, N3C_int, time, N3E_int, time, N2W_int, time, N2C_int, time, N2E_int, time, N1W_int, time, N1C_int)
    plt.xlabel('Time(s)')
    plt.ylabel('Current (A)')
    plt.legend(['S3W Current, Peak = %.2f' %peaks_0[0],'S3C Current, Peak = %.2f' %peaks_0[1],
                'S3E Current, Peak = %.2f' %peaks_0[2], 'S2W Current, Peak = %.2f' %peaks_0[3],
                'S2C Current, Peak = %.2f' %peaks_0[4], 'S2E Current, Peak = %.2f' %peaks_0[5],
                'S1W Current, Peak = %.2f' %peaks_0[6], 'S1C Current, Peak = %.2f' %peaks_0[7]])
   
plt.show()
  

    

 