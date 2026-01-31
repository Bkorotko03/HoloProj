# More plots and debug cells located in PertRNBH260128.ipynb

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["figure.dpi"] = 300
import numpy as np
import scipy as sp
import os
import sys
import datetime
import json
import warnings

# time for housekeeping and filepath setup
date = datetime.date.today()
now = datetime.datetime.now()
fdate = date.strftime('%y%m%d')
fnow = now.strftime('%y%m%d_%H%M%S')

fpath = f'./{fdate}_out'
os.makedirs(fpath,exist_ok=True)

print('Mutual information calculation for shocked Reissner-Nordstrom AdS black hole.')

# lets  set some basic params here
rmax = 1000000000
num = 500
r0min = 0.001

# now lets set some physical params from the metric
# these should become interactive at some point
d = 2
k = 0
l = 1
q = 0

# did not want to deal with handling inputs so copilot to the rescue lol
def _get_int(prompt, default, min_value=None):
    raw = input(prompt).strip()
    if raw == "":
        return default
    try:
        val = int(raw)
        if min_value is not None and val < min_value:
            print(f"Using default {default} (value must be >= {min_value}).")
            return default
        return val
    except ValueError:
        print(f"Using default {default} (invalid integer).")
        return default

def _get_float(prompt, default, min_value=None):
    raw = input(prompt).strip()
    if raw == "":
        return default
    try:
        val = float(raw)
        if min_value is not None and val < min_value:
            print(f"Using default {default} (value must be >= {min_value}).")
            return default
        return val
    except ValueError:
        print(f"Using default {default} (invalid number).")
        return default

def _get_str(prompt, default):
    raw = input(prompt).strip()
    if raw == "":
        return default
    elif (raw == "lin") or (raw == "log"):
        return raw
    else:
        print('Input lin or log')
        sys.exit()

def _get_bool(prompt, default=True):
    raw = input(prompt).strip().lower()
    if raw == "":
        return default
    elif raw in ("y", "yes"):
        return True
    elif raw in ("n", "no"):
        return False
    else:
        print("Input y or n.")
        sys.exit()

suppress_warnings = _get_bool("Suppress runtime warnings? [y/n] (default y): ", default=True)
if suppress_warnings:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

d = _get_int(f"Dimension #? (press return for default value d = {d}): ", d, min_value=2)
k = _get_float(f"k? (press return for default value k = {k}): ", k)
l = _get_float(f"l? (press return for default value l = {l}): ", l)
q = _get_float(f"q? (press return for default value q = {q}): ", q)


# function definition time. these are all some flavor of EQs found in 1405.7365
# note that i need to handle expections better

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
    rgrid1 = np.logspace(np.log10(R+0.1),np.log10(rmax-1),num)
    divAreaInteg = divArea(rgrid1,R)
    for i in range(len(r0grid)):
        r0 = r0grid[i]
        bigrgrid2 = np.linspace(r0,R-0.1,num)
        detmask = SAdeterm(bigrgrid2,r0,R) > 0.0001
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
        rarr = np.logspace(np.log10(rmin+0.1),np.log10(rmax-1),num)
        integ = 2*l/(rarr*np.sqrt(f(rarr,R)*((rarr/rmin)**(2*d-2))-f(rarr,R)))
        L = np.trapezoid(integ,rarr)
        if np.isfinite(L):
            Larr.append(L)
        else:
            Larr.append(0)
    return np.array(Larr)

def unAreaInt(rminarr,R): # area for one unbroken surface
    arealist = []
    for rmin in rminarr:
        r = np.logspace(np.log10(rmin+1),np.log10(rmax-1),num)
        integ = 2 * (r**(d-2))/np.sqrt(f(r,R)-f(r,R)*(((rmin-0.01)/r)**(2*d-2)))
        divInteg = 2 * (r**(d-2))/np.sqrt(f(r,R))
        newint = integ-divInteg
        area = np.trapezoid(newint,r)
        arealist.append(area)
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


# functions to generate the actual plots
def genAlphaPlot(Rmin,Rmax,Rnum,Rtyp='lin'):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        return 0
    
    for R in Rgrid:
        r0grid = np.linspace(r0min,0.999*R,num)
        alphagrid = alpha(r0grid,R)
        mask = k3(r0grid,R) > 0
        plt.plot(r0grid[mask],alphagrid[mask],label=f'R = {R:.3f}')

    plt.xlabel(r'$r_0$')
    plt.ylabel(r'$\alpha$')
    # plt.vlines(R,alphagrid.min(),alphagrid.max(),colors='r',label="Horizon")
    plt.semilogy()
    plt.legend()
    plt.savefig(f'{fpath}/alphavsr0.png')
    plt.close()

    # for plotting of k functions at different temperatures, uncomment. makes many a plot, not reccommended
    # k1grid = k1(r0grid,R)
    # k2grid = k2(r0grid,R)
    # k3grid = k3(r0grid,R)

    # plt.plot(r0grid,k1grid,label='k1')
    # plt.plot(r0grid,k2grid,label='k2')
    # plt.plot(r0grid,k3grid,label='k3')
    # plt.xlabel('r0')
    # plt.ylabel('k_i')
    # plt.legend()
    # # plt.semilogx()
    # plt.semilogy()
    # plt.savefig(f'{fpath}/kivsr0.png')

def genShockPlot(Rmin,Rmax,Rnum,Rtyp='lin'):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        return 0
    
    for R in Rgrid:
        r0grid = np.logspace(np.log10(r0min),np.log10(0.999*R),num)
        alphagrid = alpha(r0grid,R)
        areaInt = shockArea(r0grid,R)
        mask = areaInt > 0
        plt.plot(alphagrid[mask],areaInt[mask],label=f'R = {R:.3f}')
    
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'Area$_{A \cup B}$')
    plt.legend()
    plt.semilogx()
    plt.semilogy()
    plt.savefig(f'{fpath}/shockarrvsalpha.png')
    plt.close()

def genLvsrminPlot(Rmin,Rmax,Rnum,Rtyp='lin',rminmax=rmax):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        return 0
    
    for  R in Rgrid:
        rminarr = np.logspace(np.log10(R+0.1),np.log10(rmax-1),num)
        LvsrminArr = Lvsrmin(rminarr,R)
        plt.plot(rminarr,LvsrminArr,label=f'R = {R:.3f}')
    
    plt.xlabel(r'$r_{min}$')
    plt.ylabel(r'$L$')
    plt.xlim((2,rminmax))
    plt.legend()
    plt.semilogx()
    # plt.semilogy()
    plt.savefig(f'{fpath}/Lvsrmin.png')
    plt.close()

def genUnAreaPlot(Rmin,Rmax,Rnum,Rtyp='lin'):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        return 0
    
    for  R in Rgrid:
        rminarr = np.logspace(np.log10(R+0.1),np.log10(rmax-1),num)
        LArr = Lvsrmin(rminarr,R)
        unArea = unAreaInt(rminarr,R)
        plt.plot(LArr,unArea,label=f'R = {R:.3f}')
    
    plt.xlabel(r'$L$')
    plt.ylabel(r'Area$_{A}$')
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.savefig(f'{fpath}/unshockareavsL.png')
    plt.close()

def genMutInfPlot(Rmin,Rmax,Rnum,Rtyp='lin',Lidx=(num/2)):
    # can only give general shape of mutual information, as the "y" location is determined by the width of regions A and B
    # we will just pick a random index for now as default
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        return 0
    
    maxval = 0
    xmax = 0
    xmin = np.inf

    Lidx = np.array(int(Lidx))

    for R in Rgrid:
        r0grid = np.logspace(np.log10(r0min),np.log10(0.999*R),num)
        rminarr = np.logspace(np.log10(R+0.1),np.log10(rmax-1),num)
        alphagrid = alpha(r0grid,R)
        unArea = unAreaInt(rminarr,R)
        areaInt = shockArea(r0grid,R)
        mask = areaInt>0
        mutInf = 2*unArea[Lidx] - areaInt
        plt.plot(alphagrid[mask],mutInf[mask],label=f'R = {R:.3f}')
        
        if mutInf[mask].max() > maxval:
            maxval = mutInf[mask].max()
        if alphagrid[np.argmin(np.abs(mutInf))] > xmax:
            xmax = alphagrid[np.argmin(np.abs(mutInf))]
        if alphagrid[mask].min() < xmin:
            xmin = alphagrid[mask].min()

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\propto I(A,B)$')
    plt.ylim((0,maxval))
    plt.xlim((xmin,xmax))
    plt.legend()
    plt.semilogx()
    # plt.semilogy()
    plt.savefig(f'{fpath}/mutinfvsalpha.png')
    plt.close()

def genLCritPlot(Rmin,Rmax,Rnum,Rtyp='lin',almin=1,almax=1000): #change bounds of plot here
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        return 0
    
    maxval = 0
    xmax = 0
    xmin = np.inf

    for R in Rgrid:
        r0grid = np.logspace(np.log10(r0min),np.log10(0.999*R),num)
        rminarr = np.logspace(np.log10(R+0.1),np.log10(rmax-1),num)
        LArr = Lvsrmin(rminarr,R)
        alphagrid = alpha(r0grid,R)
        # print(len(alphagrid))
        k3grid = k3(r0grid,R)
        alphamask = k3grid > 0.1
        r0grid = r0grid[alphamask]
        alphagrid = alphagrid[alphamask]
        # print(len(alphagrid))
        unArea = unAreaInt(rminarr,R)
        areaInt = shockArea(r0grid,R)
        # print(len(areaInt))
        mask = areaInt > 0
        areaInt = areaInt[mask]
        alphagrid = alphagrid[mask]
        LCArr = LCritFunc(LArr,alphagrid,unArea,areaInt)
        
        # lets mask off more flat bits
        diff = np.diff(LCArr)
        flatmask = diff > 0
        # print(flatmask)
        flatmask = np.append(flatmask,False)
        plt.plot(alphagrid[flatmask],LCArr[flatmask],label=f'R = {R:.3f}')

        if LCArr[flatmask].max() > maxval:
            maxval = LCArr[flatmask].max()
        if alphagrid[np.argmin(np.abs(LCArr[flatmask]))] > xmax:
            xmax = alphagrid[np.argmin(np.abs(LCArr[flatmask]))]
        if alphagrid[flatmask].min() < xmin:
            xmin = alphagrid[flatmask].min()
    
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$L_{crit.}$')
    plt.legend()
    plt.semilogx()
    plt.ylim((0,maxval))
    plt.xlim((xmin,xmax))
    # plt.semilogy()
    plt.savefig(f'{fpath}/Lcritvsalpha.png')
    plt.close()

# time for temp dependence
Rmin = 2
Rmax = 5
Rnum = 3
Rlinlog = 'lin'

Rmin = _get_float(f"Minimum horizon radius? (press return for default value Rmin = {Rmin}): ", Rmin)
Rmax = _get_float(f"Maximum horizon radius? (press return for default value Rmax = {Rmax}): ", Rmax)
Rnum = _get_int(f"Numer of R values? (press return for default value Rnum = {Rnum}): ", Rnum)
Rlinglog = _get_str(f"R value lin/log spacing? (press return for default = {Rlinlog}): ", Rlinlog)

# time to make plots
genAlphaPlot(Rmin,Rmax,Rnum,Rlinlog)
genShockPlot(Rmin,Rmax,Rnum,Rlinlog)
genLvsrminPlot(Rmin,Rmax,Rnum,Rlinlog,rminmax=1000) # can change plot bounds here
genUnAreaPlot(Rmin,Rmax,Rnum,Rlinlog)
genMutInfPlot(Rmin,Rmax,Rnum,Rlinlog,Lidx=10)
genLCritPlot(Rmin,Rmax,Rnum,Rlinlog)

# json time
bigdict = {
    'rmax': rmax,
    'num': num,
    'r0min': r0min,
    'dim': d,
    'k': k,
    'q': q,
    'l': l,
    'Rmin': Rmin,
    'Rmax': Rmax,
    'Rnum': Rnum,
    'Rlinlog': Rlinlog,
}

with open(f"{fpath}/{fnow}_data.json", "w") as json_file:
    json.dump(bigdict, json_file, indent=4)

print(f'All plots and input params saved to {fpath} \ngoodbye :3')
sys.exit()
