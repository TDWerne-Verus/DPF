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

# plot settings
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'lines.linewidth': 5})

# variables
tscale = 1
sensitivity = 0.676301041736064e-9  # 0.217*1e-9; #V/(A/s)
pico_0_sense = [6.56E-10,
                7.32E-10,
                1.17E-09,
                9.07E-10,
                9.10E-10,
                9.21E-10,
                7.16E-10,
                1.15E-09,
                9.46E-10]

time = [];
raw = [];
intRog = [];
count = 0

peak_N3W = []
peak_N3C = []
peak_N3E = []
peak_N2W = []
peak_N2C = []
peak_N2E = []
peak_N1W = []
peak_N1C = []

# import csv files from folder
path = os.getcwd();  # currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"));  # files in directory

# Checksum function
checksum(csv_files, path)
# End of checksum function

# loop over the list of csv files
for file in csv_files:
    # read the csv file
    data = pd.read_csv(file)
    count = count + 1
    print(file)

    # setting variables to current channels
    df = pd.DataFrame(data, columns=['index', 'time', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

    time = df['time']

    # pico scope 0
    N3W = df['A']  # I
    N3C = df['B']  # G
    N3E = df['C']  # A
    N2W = df['D']  # C
    N2C = df['E']  # D
    N2E = df['F']  # E
    N1W = -df['G']  # P
    N1C = df['F']  # F

    # Adding 40dB attenuation to files, all signals should be 50 Ohm terminated
    # pico scope 0
    # plt.plot(time, N3W, time, N3C, time, N3E, time, N2W, time, N2C, time, N2E, time, N1W, time, N1C)
    # plt.legend(['N3W','N3C','N3E','N2W','N2C','N2E','N1W','N1C'])

    N3W = att(N3W, 40)
    N3C = att(N3C, 40)
    N3E = att(N3E, 40)
    N2W = att(N2W, 40)
    N2C = att(N2C, 40)
    N2E = att(N2E, 40)
    N1W = att(N1W, 40)
    N1C = att(N1C, 40)

    # Integrate filtered data to get current
    N3W_int = integration(N3W, pico_0_sense[0], time, tscale)
    N3C_int = integration(N3C, pico_0_sense[1], time, tscale)
    N3E_int = integration(N3E, pico_0_sense[2], time, tscale)
    N2W_int = integration(N2W, pico_0_sense[3], time, tscale)
    N2C_int = integration(N2C, pico_0_sense[4], time, tscale)
    N2E_int = integration(N2E, pico_0_sense[5], time, tscale)
    N1W_int = integration(N1W, pico_0_sense[6], time, tscale)
    N1C_int = integration(N1C, pico_0_sense[7], time, tscale)

    # Print out the max of each current plot and total
    peaks_0 = curr_peaks(N3W_int, N3C_int, N3E_int, N2W_int, N2C_int, N2E_int, N1W_int, N1C_int)

    # Creates list of currents for each module for a full day's shot series
    peak_N3W.append(peaks_0[0])
    peak_N3C.append(peaks_0[1])
    peak_N3E.append(peaks_0[2])
    peak_N2W.append(peaks_0[3])
    peak_N2C.append(peaks_0[4])
    peak_N2E.append(peaks_0[5])
    peak_N1W.append(peaks_0[6])
    peak_N1C.append(peaks_0[7])

    # Plot integrated Rogowski coil as current
    plt.figure(file)
    plt.grid(True)
    plt.xlim(0, 2e-5)
    plt.plot(time, N3W_int, time, N3C_int, time, N3E_int, time, N2W_int, time, N2C_int, time, N2E_int, time, N1W_int,
             time, N1C_int)
    plt.xlabel('Time(s)')
    plt.ylabel('Current (A/s)')
    plt.legend(['N3W Current, Peak = %.2f' % peaks_0[0], 'N3C Current, Peak = %.2f' % peaks_0[1],
                'N3E Current, Peak = %.2f' % peaks_0[2], 'N2W Current, Peak = %.2f' % peaks_0[3],
                'N2C Current, Peak = %.2f' % peaks_0[4], 'N2E Current, Peak = %.2f' % peaks_0[5],
                'N1W Current, Peak = %.2f' % peaks_0[6], 'N1C Current, Peak = %.2f' % peaks_0[7]])

plt.show()
