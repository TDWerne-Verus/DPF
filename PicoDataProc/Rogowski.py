"""
@Tiffany Berntsen - for Verus Research Magnetron Magnetic Field Calculations
In this version I am calculating the Hull cutoff and Buneman-Hartree conditions
"""

import numpy as np

mu = 4*np.pi*1e-7;                          #mu not
rt = 5.7e-6;                                #rise time
max_I = 4e6;                                #4MA for no pinch failure maximum read current

di_r = (0.00089/2) + (0.00295/2);           #radius of coax in meters
A = np.pi * np.square(di_r);                #area of the dielectric wrapped by a negligible wire radius using 36 guage
r0 = 0.02832/2 + 0.0072;                    #area of the coil around the detected current in meters 

N = 2;

V = ((mu*A*N)/(2*np.pi*r0))*(max_I/rt)
print('Voltage at Oscope', V)

N = (2*np.pi*r0*V)/(mu*A*(max_I/rt))
print('Number of turns',N)

#sensitivity
K = (mu*A*N)/(2*np.pi*r0)
print('Sensitivity',K)
print('For every A/ns we have 0.217V')
print(A)