"""
Tyler Werne - for Verus Research Rogowski Coil calibration for the data
V1, the Rogowski coil data is plotted and integrated. Coil of a particular rogowski is combined for analysis.
This is done for both the wooden and Al header. Then the averages are shown.
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

from numpy import diff
import scipy.special as special
from scipy.integrate import simps
from numpy import trapz
import datetime
from ProcessingFunctions import *
import os
from matplotlib import pyplot as plt
import string
import sklearn
from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TheilSenRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import openpyxl
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import time-frequency functions
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time

# Import utilities for loading and plotting data
from neurodsp.utils import create_times
from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.time_series import plot_time_series, plot_instantaneous_measure

from scipy.fft import fft, fftfreq, fftshift

# plot settings
plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'lines.linewidth': 5})

# Global variables
sensitivity = 1  # 67.6301041736064e-9#0.217*1e-9; #V/(A/s)
sense_array = []
sense_array2 = []
sense_array3 = []
sense_array4 = []
pear_array = []
int_array = []
int_array2 = []
int_array3 = []
int_array4 = []
pass_array = []

count = 0
pear_sense = 0.1  # pearson sensativity is 0.1 V/A
att_fact_pear = 40.0  # attinuation factor of pearson
load = 2  # dependant on what the pearson is expecting
rise_time = 5.6 * 1e-6  # typical rise time of signal
R = 50  # resistance in calibration system

# Oscope scaling
pscale = 0.5
rvscale = 2
rascale = 0.05

# import csv files from folder
path = os.getcwd()  # currently working directory
csv_files = glob.glob(os.path.join(path, "*.csv"))  # files in directory

# Checksum function
checksum(csv_files, path)
# End of checksum function

# loop over the list of csv files
pass_rogo = []
rogo = []
pear_curr = []
plt.close('all')
