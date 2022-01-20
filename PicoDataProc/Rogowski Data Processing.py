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
from scipy.integrate import quad

#Function for integration
#def integrate(x):
   # return [np.sin(x), np.exp(x)]

#function to output current distribution and total
def distribution(a, b, c):
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
    print('File: ', file)

    #setting variables to current channels
    df = pd.DataFrame(data, columns= ['index','time','A','B','C','E','H'])

    time = df['time']
    east = df['A']
    int_A = df['B']
    dead = df['C']
    center = df['E']
    center = -center
    west = df['H']

    east_curr = integration(east)

    #plot column arrays
    plt.figure()
    plt.grid(True)
    plt.xlim(0,2e-5)
    plt.plot(time, east, time, center, time, west, time, east_curr)
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage(mV)')

    #Finds distirubution of current data
    distribution(east, center, west)

    plt.legend(['East Current','Center Current','West Current'])


plt.show()
    

  

    

 