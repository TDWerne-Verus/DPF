

import csv
import numpy as np

import matplotlib.pyplot as plt
import scipy.optimize


fname = "C:\ACCSENSEVersaLog\SiteView\Download\TempData_06022022_Overnight.csv"

##====================================================================

##fields = []
##with open(fname,'r') as csvfile:
##
##    TCdat = csv.reader(csvfile)
##
##    fields = next(TCdat)
##    D = np.ndarray((np.size(fields)-1,1))
##    for Di in TCdat:
##        np.append(D,Di[1:],axis=0)

D = np.loadtxt(fname,skiprows = 1,delimiter = ',',usecols = (1,2,3))

##====================================================================

hz = 1 ##sampling rate / second
dt = 1 / hz

threshold1 = 3.5
threshold2 = 8
twindow = 120 ##seconds
tlim = 19600##12*20*60 ##time limit for shot data

l = np.shape(D)[0]
t = np.linspace(0,l*dt,l)
d1 = D[:,0]
##d2 = D[:,1]
d3 = D[:,2] ###this one is the 'good' trace that's being used below

plt.figure(1)
plt.plot(d1)
##plt.plot(d2)
plt.plot(t,d3)
plt.ylim((20,100))
plt.show()

d3 -= d1

##====================================================================
##instantaneous derivative

d3del = np.roll(d3,-1)-np.roll(d3,1)
plt.figure(2)
plt.plot(d3del)
plt.ylim((-20,20))

##====================================================================
##logical array flagging places where derivative exceeds threshold

indtrans1 = (np.abs(d3del)>threshold1)
##indtrans2 = (np.abs(d3del)>threshold2)

##smaller array of just the transient amplitudes and times
deltrans = d3del[indtrans1]
ttrans = t[indtrans1]

##plt.figure(3)
##plt.plot(ttrans,deltrans,'k.')
##plt.ylim((-20,20))
##plt.show()

##====================================================================
##two even smaller arrays that just mark the beginning of the transient
tinds = np.zeros(np.size(t))
for j in np.arange(np.size(d3del)):
    if (d3del[j] > threshold1)\
       and (d3del[j-1] < threshold1):
        tinds[j] = 1


tmark = t[tinds==1]
d3mark = d3del[tinds==1]


##plt.figure(4)
##plt.plot(tmark,d3del[tinds==1],'k.')
##plt.ylim((-20,20))
##plt.show()

##====================================================================
##instantaneous derivatives and times for times of max gradient

tmax = np.zeros(np.size(tmark[tmark<tlim]))
dmax = tmax*0
ttemp = t[t<tlim]
d3deltemp = d3del[t<tlim]

for j in np.arange(np.size(tmark[tmark<tlim])):
    lo = tmark[j]-twindow/2
    hi = tmark[j]+twindow/2
    dmax[j] = np.max(d3del[np.logical_and(t>lo, t<hi)])
    tmax[j] = ttemp[d3deltemp==dmax[j]]

plt.figure(5)
plt.plot(tmax,dmax,'k.')
##plt.show()

##====================================================================
##total height of the upward transient

smin = dmax*0
smax = smin*0

for j in np.arange(np.size(tmax)):
    lo = tmark[j]-twindow/2
    hi = tmark[j]+twindow/2
    smin[j] = np.min(d3[np.logical_and(t>lo, t<hi)])
    smax[j] = np.max(d3[np.logical_and(t>lo, t<hi)])
sdel = smax-smin

##plt.figure(6)
##plt.plot(tmax,sdel,'k.')
##plt.plot(tmax,smin,'b.')
##plt.plot(tmax,smax,'g.')

##plt.show()


##====================================================================
##====================================================================
## start working on fitting
##====================================================================
## start with conditional averaging of the 30 and 45 kV pulses
##====================================================================

n30s = np.floor(np.size(tmax)/2).astype(int)
n45s = np.floor(np.size(tmax)/2).astype(int)

tmaxs = np.append(tmax,tlim)
dshot = (np.roll(tmaxs,-1)-tmaxs)[:-1]

lmax = np.max(dshot)
nl = np.round(lmax*hz).astype(int)

TCpulse30 = np.ndarray((n30s,nl))
tpulse30 = np.arange(nl)/hz


shift = -10

for j in np.arange(n30s):
    TCpulse30[j,:] = d3[np.logical_and(t>(tmaxs[2*j]+shift),(t<tmaxs[2*j]+shift+lmax))]

##plt.figure(7)
##for j in np.arange(n30s):
##    plt.plot(tpulse30,TCpulse30[j,:])

sdel30 = sdel[0::2]
smin30 = smin[0::2]
smax30 = smax[0::2]
dshot30 = dshot[0::2]

sdel45 = sdel[1::2]

TCpulse30norm = ((TCpulse30.T-smin30)/sdel30)

##plt.figure(8)
##for j in np.arange(n30s):
##    plt.plot(tpulse30,TCpulse30norm[:,j],label = j)
##plt.xlim((0,np.min(dshot30)))
##plt.ylim((0,1.2))
##plt.legend()

##====================================================================
##====================================================================
## fit the "cooling rate" from the tail data.


ttailstart = 25000

troom = 0#25.6

dtail = d3[t>ttailstart]
ttail = t[t>ttailstart]
ttail -= ttail[0]
dtailn = (dtail)/dtail[0]

##def AexpBC(t,A,B,C):
##    return A*np.exp(-t*B)+C
##
##P,pcov = scipy.optimize.curve_fit(AexpBC,ttail,dtail,p0=(40,10,20))

def AexpBC(X):
    return X[0]*np.exp(-ttail*X[1])+troom

def chi2(X):
    return np.sum((dtail-AexpBC(X))**2)

P = scipy.optimize.minimize(chi2,[40,1e-5],\
                            method='Powell')#'Nelder-Mead')
                            #bounds=[(10,80),(1e-7,1),(10,30)],\

##def expB(X):
##    return (dtail[0]-troom)*np.exp(-ttail*X)+troom
##
##def chi2(X):
##    return np.sum((dtail-expB(X))**2)
##
##P = scipy.optimize.minimize_scalar(chi2)

##plt.figure(9)
##plt.plot(ttail,dtail)
##plt.plot(ttail,AexpBC(P.x))
##
alpha = P.x[1]


##====================================================================
##====================================================================
## fit whole series

refdat = d3[t<tlim]
reft = t[t<tlim]

synthdata = troom*np.ones(np.size(reft))

##epsilon = 0.4 ##will be X[0]
##
##gamma = alpha*100 ##will be X[1]
##
##for j in np.arange(np.size(tmax)):
##    ttemp = reft[reft>tmax[j]]
##    synthdata[reft>tmax[j]] += sdel[j]*(1-epsilon)*np.exp(-(ttemp-tmax[j])*gamma)\
##                       + sdel[j]*epsilon*np.exp(-(ttemp-tmax[j])*alpha)
##
##
##plt.figure(10)
##plt.plot(reft,refdat)
##plt.plot(reft,synthdata)


##def gensynthdata(X):
##    synthdata = reft*0+troom
##    for j in np.arange(np.size(tmax)):
##        ttemp = reft[reft>tmax[j]]
##        synthdata[reft>tmax[j]] += sdel[j]*(1-X[0])*np.exp(-(ttemp-tmax[j])*X[1])\
##                       + sdel[j]*X[0]*np.exp(-(ttemp-tmax[j])*alpha)
##    return synthdata
##
##def costsynthfit(X):
##    return np.sum((refdat - gensynthdata(X))**2)
##
##P1 = scipy.optimize.minimize(costsynthfit,[0.4, 1e-2],\
##                            method='Nelder-Mead',\
##                            bounds=[(0.3,0.7),(1e-4,1)])
##
##
##plt.figure(11)
##plt.plot(reft,refdat)
##plt.plot(reft,gensynthdata(P1.x))
##
##
##print(P1.x)

##====================================================================
##====================================================================
## fit whole series - v2 (more sophisticated model)

def gensynthdata2(X):
    synthdata = reft*0+troom
    for j in np.arange(np.size(tmax)):
        ttemp = reft[reft>tmax[j]]
        synthdata[reft>tmax[j]] += sdel[j]*np.exp(-(ttemp-tmax[j])*X[1])\
                       + sdel[j]*X[0]*(1-np.exp(-(ttemp-tmax[j])*X[1]))*np.exp(-(ttemp-tmax[j])*alpha)
    return synthdata

def costsynthfit2(X):
    return np.sum((refdat - gensynthdata2(X))**2)

P2 = scipy.optimize.minimize(costsynthfit2,[0.4, 1e-2],\
                            method='Nelder-Mead',\
                            bounds=[(0.1,0.9),(1e-4,1)])


##plt.figure(12)
##plt.plot(reft,refdat)
##plt.plot(reft,gensynthdata2(P2.x))


print(P2.x)


print('alpha = {}'.format(P.x[1]))
print('epsilon, gamma = {}'.format(P2.x))

gamma = P2.x[1]

####====================================================================
####====================================================================
#### fit whole series - v3 (more sophisticated function - lose for loop)
##
##s = np.size(tmax)
##
##def gensynthdata3(X):
##    synthdata = np.zeros((np.size(reft),np.size(tmax)))
##    synthdata += troom/np.size(tmax)
##
##    tgrid = np.outer(reft,np.ones(np.size(tmax)))
##
##    tgriddt = tgrid - tmax
##
##    synthdata += sdel*X*np.heaviside(tgriddt,1)*np.exp(-tgriddt*gamma)
##
##    synthdata += sdel*(1-X)*np.heaviside(tgriddt,1)*np.exp(-tgriddt*alpha)
##
##    synthdata = np.sum(synthdata,1)
##                                                      
##    return synthdata
##
##def costsynthfit3(X):
##    return np.sum((refdat - gensynthdata3(X))**2)
##
##P3 = scipy.optimize.minimize(costsynthfit3,[np.ones(np.size(tmax))*0.4],\
##                            method='Powell')#'Nelder-Mead',\
####                            bounds=[(-15,15),(-15,15),(-15,15),(-15,15),\
####                                    (-15,15),(-15,15),(-15,15),(-15,15),\
####                                    (-15,15),(-15,15),(-15,15),(-15,15),\
####                                    (0.1,0.9),(0.1,0.9),(0.1,0.9),(0.1,0.9),\
####                                    (0.1,0.9),(0.1,0.9),(0.1,0.9),(0.1,0.9),\
####                                    (0.1,0.9),(0.1,0.9),(0.1,0.9),(0.1,0.9)])
##
##
##plt.figure(13)
##plt.plot(reft,refdat)
##plt.plot(reft,gensynthdata3(P3.x))
##
##
##print(P3.x)
##
##print('alpha = {}'.format(alpha))
##print('gamma = {}'.format(gamma))
##print('epsilon = {}'.format(P3.x))


##====================================================================
##====================================================================
## fit whole series - v3 (more sophisticated function - lose for loop)

s = np.size(tmax)

def gensynthdata3(X):
    synthdata = np.zeros((np.size(reft),np.size(tmax)))
    synthdata += troom/np.size(tmax)

    tgrid = np.outer(reft,np.ones(np.size(tmax)))

    tgriddt = tgrid - tmax

    synthdata += sdel*X[0:s]*np.heaviside(tgriddt,1)*np.exp(-tgriddt*gamma*X[s])

    synthdata += sdel*(1-X[0:s])*np.heaviside(tgriddt,1)*np.exp(-tgriddt*alpha*X[s+1])

    synthdata = np.sum(synthdata,1)
                                                      
    return synthdata

def costsynthfit3(X):
    return np.sum((refdat - gensynthdata3(X))**2)

guess = np.append(np.ones(s)*0.4,[1,1])

P3 = scipy.optimize.minimize(costsynthfit3,guess,\
                            method='Powell')#'Nelder-Mead',\
##                            bounds=[(-15,15),(-15,15),(-15,15),(-15,15),\
##                                    (-15,15),(-15,15),(-15,15),(-15,15),\
##                                    (-15,15),(-15,15),(-15,15),(-15,15),\
##                                    (0.1,0.9),(0.1,0.9),(0.1,0.9),(0.1,0.9),\
##                                    (0.1,0.9),(0.1,0.9),(0.1,0.9),(0.1,0.9),\
##                                    (0.1,0.9),(0.1,0.9),(0.1,0.9),(0.1,0.9)])


plt.figure(13)
plt.plot(reft,refdat)
plt.plot(reft,gensynthdata3(P3.x))


print(P3.x)

print('alpha = {}'.format(alpha*P3.x[s+1]))
print('gamma = {}'.format(gamma*P3.x[s]))
print('epsilon = {}'.format(P3.x[0:s]))


## 5/31 yield data:

yield45 = [52221, 45410, 54364, 46890, 36428, 38701]

plt.figure(14)
plt.plot(yield45,P3.x[0:s:2])




plt.show()
