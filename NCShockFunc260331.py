# This file contains the functions to generate the non-com arrays

import numpy as np
import scipy as sp

rmax = 100
num = 3000
eps = 0.0001

# time to define aux functions, derived from the Fischler paper and from 1405.7365

# define some useful functions
def f(r,R):
    return 1 - ((R/r)**4)

def fp(r,R):
    return 1 + 4*(R**4)/(r**5)

def h(r,a):
    return 1/(1 + (a**4) * (r**4))

def b(r,rmin):
    return ((r**6)/(rmin**6)) - 1

def beta(R):
    return 4 * np.pi / fp(R,R)

def gamma2(r0,R):
    return -(r0**8)*f(r0,R)

# define interior radial cutoff for imaginary dt
def determ(r,R,r0):
    return 1 + (r**8)*f(r,R)/gamma2(r0,R)

# we need to quickly define a divergent integrand for k2 because shit blows up otherwise. look at the things in goodnotes and youll be convinced that this is true:
def k2divint(r,R):
    return 1/f(r,R)

# now we want to define the big integrals which go into alpha
# there will still need to be a radial cutoff for k3...
def k1(r0grid,R):
    list = []
    for i in range(num):
        r0 = r0grid[i]
        rbar = eps # r0grid.min()/2
        rgrid = np.linspace(rbar,r0,num)
        integ = 1/f(rgrid,R)
        mask = np.isfinite(integ)
        sum = (4*np.pi/beta(R)) * np.trapezoid(integ[mask],rgrid[mask])
        list.append(sum)
    return np.array(list)

def k2(r0grid,R):
    list = []
    for i in range(num):
        r0 = r0grid[i]
        rgrid = np.logspace(np.log10(R+0.1),np.log10(rmax-eps),num)
        # print(rgrid)
        integ = ((1 - 1/((rgrid**2) * np.sqrt(determ(rgrid,R,r0))))/f(rgrid,R)) - k2divint(rgrid,R)
        # print(integ)
        mask = np.isfinite(integ)
        sum = (2*np.pi/beta(R)) * np.trapezoid(integ[mask],rgrid[mask])
        list.append(sum)
    return np.array(list)

def k3(r0grid,R):
    list = []
    for i in range(num):
        r0 = r0grid[i]
        bigrgrid = np.linspace(r0,R-0.001,num)
        detmask = determ(bigrgrid,R,r0) > 0
        if len(bigrgrid[detmask]) >= (num-2):
            bigrmin,bigrmax = bigrgrid[detmask].min(), bigrgrid[detmask].max()
            rgrid = np.linspace(bigrmin,bigrmax,num)
            integ = (1-1/((rgrid**2) * np.sqrt(determ(rgrid,R,r0))))/f(rgrid,R)
            for idx in range(len(integ)):
                if determ(rgrid[idx],R,r0) <= 0:
                    integ[idx] = 0
            mask = np.isfinite(integ)
            sum = (4*np.pi/beta(R)) * np.trapezoid(integ[mask],rgrid[mask])
            list.append(sum)
        else:
            list.append(0)
    return np.array(list)

def alpha(r0grid,R):
    return 2 * np.exp(k1(r0grid,R)+k2(r0grid,R)+k3(r0grid,R))

# note that it's important that both AInteg and divArea have the same power of r up top
# even if expressions are equivalent, numerically they scale much differently if AInteg has r**6 on top for fun computer reasons probably
def AInteg(rgrid,r0,R):
    return (rgrid**2) / np.sqrt(gamma2(r0,R)/(rgrid**8) + f(rgrid,R))

def SADeterm(r,r0,R):
    return gamma2(r0,R) + (r**8)*f(r,R)

def divArea(rgrid,R):
    return (rgrid**2) / np.sqrt(f(rgrid,R))

def shockArea(r0grid,R):
    areaList = []
    area1List = []
    area2List = []
    rgrid1 = np.logspace(np.log10(R+eps),np.log10(rmax-eps),num)
    divAreaInteg = divArea(rgrid1,R)
    for i in range(len(r0grid)):
        r0 = r0grid[i]
        bigrgrid2 = np.linspace(r0,R-eps,num)
        detmask = SADeterm(bigrgrid2,r0,R) > 0
        if len(bigrgrid2[detmask]) >= (len(bigrgrid2)-1):
            bigrmin, bigrmax = bigrgrid2[detmask].min(), bigrgrid2[detmask].max()
            rgrid2 =  np.linspace(bigrmin,bigrmax,num)
            areaInteg1 = AInteg(rgrid1,r0,R) - divAreaInteg
            area1 = np.trapezoid(areaInteg1,rgrid1)
            areaInteg2 = AInteg(rgrid2,r0,R)
            mask = np.isfinite(areaInteg2)
            area2 = np.trapezoid(areaInteg2[mask],rgrid2[mask])
            areaSum = (2 * area1) + (4 * area2)
            areaList.append(areaSum)
            area1List.append(area1)
            area2List.append(area2)
        else:
            areaList.append(0)
            area1List.append(0)
            area2List.append(0)
    return np.array(areaList) #, area1List, area2List

# time to relate L and rmin
def Lvsrmin(rminarr,R,a):
    Larr = []
    for rmin in rminarr:
        rarr = np.logspace(np.log10(rmin+eps),np.log10(rmax),num)
        integ = 2/np.sqrt((rarr**4) * f(rarr,R) * h(rarr,a) * b(rarr,rmin))
        L = np.trapezoid(integ,rarr)
        if np.isfinite(L):
            Larr.append(L)
        else:
            Larr.append(0)
    return np.array(Larr)

def rMinCutoff(rminarr,LArr):
    nCut = int(len(rminarr)*0.9) #we dont want bad solutions for the argmin way out at large r
    LarrCut = LArr[:-nCut]
    idx = LarrCut.argmin()
    rminmax = rminarr[idx]
    rminmin = rminarr.min()
    newrArr = np.logspace(np.log10(rminmin),np.log10(rminmax),num)
    return newrArr

def unAreaInt(rminarr,R,a):
    areaList = []
    bigRgrid = np.logspace(np.log10(R+eps),np.log10(rmax),num)
    # divArea1 = (2/(a**2)) * (((a**4)*(rmax**4)/4) + (1 + (a**4)*(R**4))*np.log(a * rmax)/2) # this is the more simplified taylor expand
    divArea = (1/(4 * a**2)) * (2*(a**4)*(rmax**4) + 1 - (a**4)*(R**4) + (1+(a**4)*(R**4))*np.log(4*(a**4)*(rmax**4)/(1+(a**4)*(R**4))))
    fullDivAreaInt = 2*bigRgrid/np.sqrt(f(bigRgrid,R)*h(bigRgrid,a))
    fullDivArea = np.trapezoid(fullDivAreaInt,bigRgrid)
    for rmin in rminarr:
        r = np.logspace(np.log10(rmin+eps),np.log10(rmax),num)
        integ = 2 * r * np.sqrt((1 + b(r,rmin))/(f(r,R)*h(r,a)*b(r,rmin)))

        # fullDivAreaInt = 2*r/np.sqrt(f(r,R)*h(r,a))
        # fullDivArea = np.trapezoid(fullDivAreaInt,r)
        # integ2 = 2 * r * (np.sqrt((1 + b(r,rmin))/(f(r,R)*h(r,a)*b(r,rmin))) - 1/np.sqrt(f(r,R)*h(r,a)) )
        area1 = np.trapezoid(integ,r)
        # area2 = np.trapezoid(integ2,r)
        
        # print(fullDivArea - divArea)
        areaList.append(area1 - divArea)
        # areaList.append(area2)
    return np.array(areaList)

def LCritFunc(Lgrid,alphagrid,unShockGrid,shockGrid):
    Lcritarr = []
    for i in range(len(alphagrid)):
        shockArea = shockGrid[i]
        mutinfGrid = 2 * unShockGrid - shockArea
        mutinfInterp = sp.interpolate.interp1d(mutinfGrid,Lgrid,fill_value="extrapolate")
        Lcrit = mutinfInterp(0)
        Lcritarr.append(Lcrit)
    return np.array(Lcritarr)