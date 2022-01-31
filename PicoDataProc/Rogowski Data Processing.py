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


#import csv files from folder
path = os.getcwd();                                     #currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"));      #files in directory

#plot settings
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'lines.linewidth': 5})

#variables
sensitivity = 0.217*1e-9; #V/(A/s)
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
    N3E = df['A']
    int_A = df['B'] 
    dead = df['C']
    center = df['E']
    center = -center
    west = df['H']
    '''
    east_curr = integration(east)

    east_filt = filter(east)    #filter raw data
    east_curr = integration(east_filt,time,sensitivity)
    S2C = df['D']
    S2C = -S2C      #Inverted Rogowski Coil
    N3C = df['E']
    N3C = -N3C      #Inverted Rogowski Coil
    N2C = df['F']
    N2C = -N2C      #Inverted Rogowski Coil
    N1C = df['G']
    N3W = df['H']
    
    #Adding attenuation to files
    N3E = att(N3E)
    S2C = att(S2C)
    N3C = att(N3C)
    N2C = att(N2C)
    N1C = att(N1C)
    N3W = att(N3W)

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
    peaks = curr_peaks(N3E_int, N3W_int, N1C_int, N2C_int, N3C_int, S2C_int)
            
    peak_N3E.append(peaks[0])
    peak_N3W.append(peaks[1])
    peak_N1C.append(peaks[2])
    peak_N2C.append(peaks[3])
    peak_N3C.append(peaks[4])
    peak_S2C.append(peaks[5])
    
    
    dif12 = (peaks[4]-peaks[3])/peaks[3]
    dif23 = (peaks[3]-peaks[2])/peaks[2]
    dif13 = (peaks[2]-peaks[4])/peaks[4]
 
    '''print('N1C to N2C %f' %dif12)
    print('N2C to N3C %f' %dif23)
    print('N1C to N3C %f' %dif13)'''

    
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
    plt.figure(file)
    plt.grid(True)
    plt.xlim(0,2e-5)
    #plt.plot(time,N3E_int,time,S2C_int,time,N3C_int, time, N2C_int, time, N1C_int, time, N3W_int)
    plt.plot(time, N3E_int, time, N3W_int, time, N1C_int, time, N2C_int, time, N3C_int)
    plt.xlabel('Time(s)')
    plt.ylabel('Current (A)')
    #plt.legend(['N3E = %.2f' %peaks[0] ,'S2C = %.2f' %0,'N3C = %.2f' %peaks[2] ,'N2C = %.2f' %peaks[3] ,'N1C = %.2f' %peaks[4],'N3W = %.2f' %peaks[5]])
    plt.legend(['N3E Current, Peak = %.2f' %peaks[0],'N3W Current, Peak = %.2f' %peaks[1],'N1C Current, Peak = %.2f' %peaks[2], 'N2C Current, Peak = %.2f' %peaks[3],'N3C Current, Peak = %.2f' %peaks[4]])
   
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
    '''df = DataFrame({'N3 East': peaks[0]})
    print(df)
    df.to_excel('test.xlsx',sheet_name = 'sheet1', index = False)'''

plt.figure('N3E Coil')
plt.plot(range(0,count), peak_N3E)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
var = (max(peak_N3E) - min(peak_N3E))*100 / min(peak_N3E)
print(var)
plt.legend(['Percent Variation = %6.2f' %var])

#plt.figure('S2C Coil')
#plt.plot(range(0,count), peak_S2C)
#plt.xlabel('File Index')
#plt.ylabel('Current (A)')
#var = (max(peak_S2C)-min(peak_S2C))*100 /min(peak_S2C)
#plt.legend(['Percent Variation = %6.2f' %var])

plt.figure('N3C Coil')
plt.plot(range(0,count), peak_N3C)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
var = (max(peak_N3C)-min(peak_N3C))*100 /min(peak_N3C)
plt.legend(['Percent Variation = %6.2f' %var])

plt.figure('N2C Coil')
plt.plot(range(0,count), peak_N2C)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
var = (max(peak_N2C)-min(peak_N2C))*100 /min(peak_N2C)
plt.legend(['Percent Variation = %6.2f' %var])

plt.figure('N1C Coil')
plt.plot(range(0,count), peak_N1C)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
var = (max(peak_N1C)-min(peak_N1C))*100 /min(peak_N1C)
plt.legend(['Percent Variation = %6.2f' %var])

plt.figure('N3W Coil')
plt.plot(range(0,count), peak_N3W)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
var = (max(peak_N3W)-min(peak_N3W))*100 /min(peak_N3W)
plt.legend(['Percent Variation = %6.2f' %var])

'''
plt.figure('Scaled Current Comparisons')
plt.plot(range(1,count+1), peak_N1C, range(1,count+1), peak_N2C, range(1,count+1), peak_N3C)
plt.xlabel('File Index')
plt.ylabel('Current (A)')
varN1C = (max(peak_N1C)-min(peak_N1C))*100 /min(peak_N1C)
varN2C = (max(peak_N2C)-min(peak_N2C))*100 /min(peak_N2C)
varN3C = (max(peak_N3C)-min(peak_N3C))*100 /min(peak_N3C)
N12diff = (max(peak_N1C)-max(peak_N2C))*100/ max(peak_N2C)
N23diff = (max(peak_N2C)-max(peak_N3C))*100/ max(peak_N3C)
plt.legend(['N1C','N2C','N3C'])'''


plt.show()
  

    

 