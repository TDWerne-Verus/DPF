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



#import csv files from folder
path = os.getcwd();                                     #currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"));      #files in directory

#variables
sensitivity = 0.217; #V/(A/ns)
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
    df = pd.DataFrame(data, columns= ['index','time','A','B','C','E','H'])

    time = df['time']
    east = df['A']
    int_A = df['B']
    dead = df['C']
    center = df['E']
    center = -center
    west = df['H']
    '''
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
'''

  

    

 