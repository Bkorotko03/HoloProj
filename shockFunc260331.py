# This guy is the functions from the commutative case for calculating entanglement entropy

import numpy as np
import scipy as sp

# lets  set some basic params here
rmax = 10000000
num = 3000
r0min = 0.0001
eps = 0.0001

# now lets set some physical params from the metric
d = 5
k = 0 
l = 1
q = 0

# functions from metric
def mu(R):
    return (R**(d-2))*(k+(q**2)/(R**(2*d-4))+(R**2)/(l**2))

def f(r,R):
    return k - mu(R)/(r**(d-2)) + (q**2)/(r**(2*d-4)) + (r**2)/(l**2)

def fp(r,R):
    return (d-2)*mu(R)/(r**(d-3)) + (4-2*d)*(q**2)/(r**(2*d-5))+ 2*r/(l**2)

def beta(R):
    return 4 * np.pi / fp(R,R)

def gamma2(r0,R):
    return -1*f(r0,R)*(r0**(2*d-4))

# k_i squareroot determinant
def determ(r,R,k=k,q=q,l=l):
    return k*(r**(2*d-4)) - mu(R)*(r**(d-2)) + (q**2) + (r**(2*d-2))/(l**2)

# functions for generating \alpha
def k1(r0grid,R):
    list = []
    for i in range(num):
        r0 = r0grid[i]
        rbar = R / 2
        rgrid = np.linspace(rbar,r0,num)
        integ = 1/f(rgrid,R)
        mask = np.isfinite(integ)
        sum =  (4 * np.pi / beta(R)) * np.trapezoid(integ[mask],rgrid[mask])
        list.append(sum)
    return np.array(list)

def k2(r0grid,R):
    list = []
    for i in range(num):
        r0 = r0grid[i]
        rgrid = np.logspace(np.log10(R+0.1),np.log10(rmax-0.1),num)
        integ = (1 - 1/(np.sqrt(1+f(rgrid,R)*(rgrid**(2*d-4))/(gamma2(r0,R)))))/f(rgrid,R)
        mask = np.isfinite(integ)
        sum = (2*np.pi/beta(R)) * np.trapezoid(integ[mask],rgrid[mask])
        list.append(sum)
    return np.array(list)

def k3(r0grid,R):
    list = []
    for i in range(num):
        r0 = r0grid[i]
        bigrgrid = np.linspace(r0,0.999*R,num)
        # r0det = determ(r0,R)
        # rdet = determ(bigrgrid,R)
        # print((1 - (rdet/r0det)))
        detmask = (1 + ((f(bigrgrid,R)*(bigrgrid**(2*d-4)))/(-1*f(r0,R)*(r0**(2*d-4))))) > 0
        # print(detmask)
        if len(bigrgrid[detmask]) >= (num-2):
            bigrmin, bigrmax = bigrgrid[detmask].min(), bigrgrid[detmask].max()
            rgrid =  np.linspace(bigrmin,bigrmax,num)
            # print(rgrid)
            integ = (1 - 1/(np.sqrt(1+f(rgrid,R)*(rgrid**(2*d-4))/(gamma2(r0,R)))))/f(rgrid,R)
            # print(integ)
            for idx in range(len(integ)):
                if (1+f(rgrid[idx],R)*(rgrid[idx]**(2*d-4))/(gamma2(r0,R))) <= 0:
                    integ[idx]=0
            mask = np.isfinite(integ)
            sum = (4*np.pi/beta(R)) * np.trapezoid(integ[mask],rgrid[mask])
            list.append(sum)
        else:
            list.append(0)
    return np.array(list)

def alpha(r0grid,R):
    return 2*np.exp(k1(r0grid,R)+k2(r0grid,R)+k3(r0grid,R))

# definining functions for shocked area
def AInteg(rgrid,r0,R):
    return (rgrid**(d-2)) / np.sqrt(f(rgrid,R) + gamma2(r0,R)*(rgrid**(4-2*d)))

def divArea(rgrid,R):
    return (rgrid**(d-2)) / np.sqrt(f(rgrid,R))

def SAdeterm(r,r0,R):
    return f(r,R) + gamma2(r0,R)*(r**(4-2*d))

def shockArea(r0grid,R):
    areaList = []
    # area1List = [] # this guy and below are for debug
    # area2List = []
    rgrid1 = np.logspace(np.log10(R+eps),np.log10(rmax-eps),num)
    divAreaInteg = divArea(rgrid1,R)
    for i in range(len(r0grid)):
        r0 = r0grid[i]
        bigrgrid2 = np.linspace(r0,R-eps,num)
        detmask = SAdeterm(bigrgrid2,r0,R) > eps
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
            # area1List.append(area1)
            # area2List.append(area2)
        else:
            areaList.append(0)
            # area1List.append(0)
            # area2List.append(0)
    return np.array(areaList) # , area1List, area2List


# defining functions for unbroken surfaces
def Lvsrmin(rminarr,R):
    Larr = []
    for rmin in rminarr:
        rarr = np.logspace(np.log10(rmin+eps),np.log10(rmax-eps),num)
        integ = 2*l/(rarr*np.sqrt(f(rarr,R)*((rarr/rmin)**(2*d-2))-f(rarr,R)))
        L = np.trapezoid(integ,rarr)
        if np.isfinite(L):
            Larr.append(L)
        else:
            Larr.append(0)
    return np.array(Larr)

def unAreaInt(rminarr,R): # area for one unbroken surface
    arealist = []
    Rgrid = np.logspace(np.log10(R),np.log10(rmax),num)
    for rmin in rminarr:
        r = np.logspace(np.log10(rmin+eps),np.log10(rmax-eps),num)
        integ = 2 * (r**(d-2))/np.sqrt(f(r,R)-f(r,R)*(((rmin-eps)/r)**(2*d-2)))
        divInteg = 2 * (r**(d-2))/np.sqrt(f(r,R))
        newint = integ - divInteg
        area = np.trapezoid(integ,r)
        divArea = np.trapezoid(divInteg,Rgrid)
        arealist.append(area-divArea)
    return np.array(arealist)


# compare L_crit to alpha
def LCritFunc(Lgrid,alphagrid,unShockGrid,shockGrid):
    Lcritarr = []
    for i in range(len(alphagrid)):
        shockArea = shockGrid[i]
        mutinfGrid = 2 * unShockGrid - shockArea
        mutinfInterp = sp.interpolate.interp1d(mutinfGrid,Lgrid,fill_value="extrapolate")
        Lcrit = mutinfInterp(0)
        Lcritarr.append(Lcrit)
    return np.array(Lcritarr)