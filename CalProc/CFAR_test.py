# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:59:03 2022

@author: tyler.werne

Created to test CFAR algorithm
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import scipy
from scipy.signal import find_peaks
from scipy import signal
from scipy import integrate
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import simps
from numpy import trapz
import datetime
from ProcessingFunctions import *

size = [1, 10000]
x_t = np.zeros(size)
w_t = np.random.randn(size[0], size[1])

x_t[0, 4500] = 50
s_t = x_t + w_t
s_t = s_t[0, :]
Pfa = 0.001
refLength = 50
guardLength = 1
CFARThreshold = CFAR(s_t, Pfa, refLength, guardLength)

for t in range(0, len(CFARThreshold)):
    if (s_t[t] > CFARThreshold[t]):
        start = t
        break;
