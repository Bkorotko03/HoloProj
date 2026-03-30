# interactive script to plot non-com shocked entropies

# note that implementing something like a star map instead of these evil for loops would be much better
# implement optional diagnostic print messages

import matplotlib.pyplot as plt
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["figure.dpi"] = 300
import numpy as np
import scipy as sp
import os
import sys
import datetime
import json
import warnings

date = datetime.date.today()
now = datetime.datetime.now()
fdate = date.strftime('%y%m%d')
fnow = now.strftime('%y%m%d_%H%M%S')

fpath = f'./NCOut_{fdate}'
fpathn = f'./NCNormOut_{fdate}'
os.makedirs(fpath,exist_ok=True)
os.makedirs(fpathn,exist_ok=True)

print('Entropy calculations for non-commutative shocked AdS black hole.')

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

suppress_warnings = _get_bool("Suppress runtime warnings? [y/n] (press return for default y): ", default=True)
if suppress_warnings:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

# lets set basic params and then double check with prompts
# these are UV cutoff, precision, and more precision
rmax = 100
num = 3000
eps = 0.0001

rmax = _get_int(f"Maximum r value? (press return for default value rmax = {rmax}): ", rmax)
num = _get_int(f"Array size? (press return for default value num = {num}): ", num)
eps = _get_float(f"Epsilon value? (press return for default value epsilon = {eps}): ", eps)

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

# now time to define functions to make plots
def genAlphaPlot(Rmin,Rmax,Rnum,Rtyp='lin',norm=False):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        sys.exit(1)

    # print(Rgrid)
    for R in Rgrid:
        r0grid = np.linspace(eps,R-eps,num)
        alphagrid = alpha(r0grid,R)
        mask = k3(r0grid,R) != 0
        normr0grid = r0grid/R
        if norm == True:
            normalphagrid = alphagrid / alphagrid.max()
            plt.plot(normr0grid[mask],normalphagrid[mask],label=f'R = {R:.3f}')
        elif norm == False:
            plt.plot(normr0grid[mask],alphagrid[mask],label=f'R = {R:.3f}')
        else:
            print('norm must be boolean value.')
            sys.exit(1)

    if norm == True:
        plt.xlabel(r'$r_0/R$')
        plt.ylabel(r'$\alpha / \alpha_{max}$')
        plt.semilogy()
        plt.legend()
        plt.savefig(f'{fpathn}/NCNormalphavsr0.png')
        plt.close()
    elif norm == False:
        plt.xlabel(r'$r_0/R$')
        plt.ylabel(r'$\alpha$')
        plt.semilogy()
        plt.legend()
        plt.savefig(f'{fpath}/NCalphavsr0.png')
        plt.close()
    else:
        print('norm must be boolean value.')
        sys.exit(1)

def genShockPlot(Rmin,Rmax,Rnum,Rtyp='lin',norm=False):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        sys.exit(1)
    
    # print(Rgrid)
    for R in Rgrid:
        r0grid = np.logspace(np.log10(eps),np.log10(R-eps),num)
        alphagrid = alpha(r0grid,R)
        areaInt = shockArea(r0grid,R)
        # print(areaInt)
        mask = areaInt > 0
        
        if norm == True:
            # normalphagrid = alphagrid / alphagrid.max()
            normareaInt = areaInt/areaInt.max()
            plt.plot(alphagrid[mask],normareaInt[mask],label = f'R = {R:.3f}')
        elif norm == False:
            plt.plot(alphagrid[mask],areaInt[mask],label = f'R = {R:.3f}')
        else:
            print('norm must be boolean value.')
            sys.exit(1)

    if norm == True:
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'Area$_{A \cup B} / $Area$_{max}$')
        plt.legend()
        plt.semilogx()
        plt.savefig(f'{fpathn}/NCNormshockarrvsalpha.png')
        plt.close()
    elif norm == False:
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'Area$_{A \cup B}$')
        plt.semilogy()
        plt.legend()
        plt.semilogx()
        plt.savefig(f'{fpath}/NCshockarrvsalpha.png')
        plt.close()
    else:
        print('norm must be boolean value.')
        sys.exit(1)

def genLvsrminPlot(Rmin,Rmax,Rnum,a=0.1,Rtyp='lin',rminmax=rmax):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        sys.exit(1)
    
    for R in Rgrid:
        rminarr = np.logspace(np.log10(R+eps),np.log10(rmax),num)
        LvsrminArr = Lvsrmin(rminarr,R,a)
        normrminarr = rminarr/R
        plt.plot(normrminarr,LvsrminArr,label=f'R = {R:.3f}')

    plt.xlabel(r'$r_{min}/R$')
    plt.ylabel(r'$L$')
    # plt.xlim((1/2,10))
    plt.legend()
    plt.semilogx()
    plt.savefig(f'{fpath}/NCLvsrmin_a{a:.3f}.png')
    plt.close()

def genAdjLvsrminPlot(Rmin,Rmax,Rnum,a=0.1,Rtyp='lin'):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        sys.exit(1)
    
    for R in Rgrid:
        rminarr = np.logspace(np.log10(R+eps),np.log10(rmax),num)
        LvsrminArr = Lvsrmin(rminarr,R,a)
        rminarr2 = rMinCutoff(rminarr,LvsrminArr)
        LvsrminArr2 = Lvsrmin(rminarr2,R,a)
        normrminarr = rminarr2/R
        plt.plot(normrminarr,LvsrminArr2,label = f'R = {R:.3f}')

    plt.xlabel(r'$r_{min}/R$')
    plt.ylabel(r'$L$')
    # plt.semilogx()
    # plt.semilogy()
    plt.legend()
    plt.savefig(f'{fpath}/NCLvsrminCut_a{a:.3f}.png')
    plt.close()

def genUnAreaPlot(Rmin,Rmax,Rnum,a=0.1,Rtyp='lin',norm=False):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        sys.exit(1)
    
    for R in Rgrid:
        rminarr = np.logspace(np.log10(R+eps),np.log10(rmax),num)
        LvsrminArr = Lvsrmin(rminarr,R,a)
        rminarr2 = rMinCutoff(rminarr,LvsrminArr)
        LvsrminArr2 = Lvsrmin(rminarr2,R,a)
        unArea = unAreaInt(rminarr2,R,a)
        if norm == True:
            normunArea = unArea/unArea.max()
            plt.plot(LvsrminArr2,normunArea,label=f'R = {R:.3f}')
        elif norm == False:
            plt.plot(LvsrminArr2,unArea,label=f'R = {R:.3f}')
        else:
            print('norm must be boolean value.')
            sys.exit(1)

    if norm == True:
        plt.xlabel(r'$L$')
        plt.ylabel(r'Area$_{A} / $Area$_{max}$')
        plt.legend()
        plt.savefig(f'{fpathn}/NCNormunshockareavsL_a{a:.3f}.png')
        plt.close()
    elif norm == False:
        plt.xlabel(r'$L$')
        plt.ylabel(r'Area$_{A}$')
        plt.semilogy()
        plt.legend()
        plt.savefig(f'{fpath}/NCunshockareavsL_a{a:.3f}.png')
        plt.close()
    else:
        print('norm must be boolean value.')
        sys.exit(1)

def genMutInfPlot(Rmin,Rmax,Rnum,a=0.1,Rtyp='lin',Lidx=(num/2),norm=False):
    # can only give general shape of mutual information, as the "y" location is determined by the width of regions A and B
    # we will just pick a random index for now as default
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        sys.exit(1)
    
    maxval = 0; xmax = 0; xmin = np.inf
    Lidx = np.array(int(Lidx))

    for R in Rgrid:
        r0grid = np.logspace(np.log10(eps),np.log10(R-eps),num)
        rminarr = np.logspace(np.log10(R+eps),np.log10(rmax),num)
        LvsrminArr = Lvsrmin(rminarr,R,a)
        rminarr2 = rMinCutoff(rminarr,LvsrminArr)
        # LvsrminArr2 = Lvsrmin(rminarr2,R,a)
        unArea = unAreaInt(rminarr2,R,a)
        alphagrid = alpha(r0grid,R)
        areaInt = shockArea(r0grid,R)
        # print(areaInt)
        mask = areaInt > 0
        mutInf = 2*unArea[Lidx] - areaInt

        if mutInf[mask].max() > maxval:
            maxval = mutInf[mask].max()
        if alphagrid[np.argmin(np.abs(mutInf))] > xmax:
            xmax = alphagrid[np.argmin(np.abs(mutInf))]
        if alphagrid[mask].min() < xmin:
            xmin = alphagrid[mask].min()
        
        if norm == True:
            normmutInf = mutInf[mask]/mutInf[mask].max()
            normalphagrid = alphagrid[mask]/alphagrid[mask].max()
            plt.plot(normalphagrid,normmutInf,label=f'R = {R:.3f}')
            if normalphagrid.min() < xmin:
                xmin = normalphagrid.min()
        elif norm == False:
            plt.plot(alphagrid[mask],mutInf[mask],label=f'R = {R:.3f}')
            # plt.semilogy()
        else:
            print('norm must be boolean value.')
            sys.exit(1)

    if norm == True:
        plt.xlabel(r'$\alpha / \alpha_{max}$')
        plt.ylabel(r'$\propto I(A,B) / I(A,B)_{max}$')

        plt.ylim((xmin,1))
        plt.xlim((xmin,1))
        plt.legend()
        plt.semilogx()
        plt.savefig(f'{fpathn}/NCNormmutinfvsalpha_a{a:.3f}.png')
        plt.close()
    elif norm == False:
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\propto I(A,B)$')
        plt.semilogy()
        plt.ylim((eps,maxval))
        plt.xlim((xmin,xmax))
        plt.semilogx()
        plt.legend()
        plt.savefig(f'{fpath}/NCmutinfvsalpha_a{a:.3f}.png')
        plt.close()
    else:
        print('norm must be boolean value.')
        sys.exit(1)

def genLCritPlot(Rmin,Rmax,Rnum,a=0.1,Rtyp='lin',norm=False):
    if Rtyp == 'lin':
        Rgrid = np.linspace(Rmin,Rmax,Rnum)
    elif Rtyp == 'log':
        Rgrid = np.logspace(np.log10(Rmin),np.log10(Rmax),Rnum)
    else:
        print('Invalid Rtyp, use lin or log')
        return 0
    
    for R in Rgrid:
        r0grid = np.logspace(np.log10(eps),np.log10(R-eps),num)
        rminarr = np.logspace(np.log10(R+eps),np.log10(rmax),num)
        LvsrminArr = Lvsrmin(rminarr,R,a)
        rminarr2 = rMinCutoff(rminarr,LvsrminArr)
        LvsrminArr2 = Lvsrmin(rminarr2,R,a)
        alphagrid = alpha(r0grid,R)
        # print(len(alphagrid))
        k3grid = k3(r0grid,R)
        alphamask = k3grid != 0
        r0grid = r0grid[alphamask]
        alphagrid = alphagrid[alphamask]
        # print(len(alphagrid))

        unArea = unAreaInt(rminarr2,R,a)
        areaInt = shockArea(r0grid,R)
        # print(f'un:{len(unArea)}, sh:{len(areaInt)}')

        mask = areaInt > 0
        areaInt = areaInt[mask]
        alphagrid = alphagrid[mask]
        # print(f'un:{len(unArea)}, sh:{len(areaInt)}')
        LCArr = LCritFunc(LvsrminArr2,alphagrid,unArea,areaInt)
        # print(LCArr)

        diff = np.diff(LCArr)
        flatmask = diff > 0
        # print(flatmask)

        # alphagrid = alphagrid[flatmask]
        # LCArr = LCArr[flatmask]

        posmask = LCArr > 0
        alphagrid = alphagrid[posmask]
        LCArr = LCArr[posmask]

        flatmask = np.append(flatmask,False)

        if norm == True:
            normLCArr = LCArr / LCArr.max()
            normalphagrid = alphagrid/alphagrid.max()
            plt.plot(normalphagrid,normLCArr,label=f'R = {R:.3f}')
        elif norm == False:
            plt.plot(alphagrid,LCArr,label=f'R = {R:.3f}')
        else:
            print('norm must be boolean value.')
            sys.exit(1)

    if norm == True:
        plt.xlabel(r'$\alpha / \alpha_{max}$')
        plt.ylabel(r'$L_{crit} / L_{crit,max}$')
        plt.legend()
        plt.savefig(f'{fpathn}/NCNormLcritvsalpha_a{a:.3f}.png')
        plt.close()
    elif norm == False:
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$L_{crit}$')
        plt.semilogx()
        plt.semilogy()
        plt.legend()
        plt.savefig(f'{fpath}/NCLcritvsalpha_a{a:.3f}.png')
        plt.close()
    else:
        print('norm must be boolean value.')
        sys.exit(1)

# now its time to do things fr

# make default values for temps and NC param
Rmin = 1
Rmax = 5
Rnum = 3
Rlinlog = 'lin'

amin = 0.1
amax = 0.5
anum = 3

Rmin = _get_float(f"Minimum horizon radius? (press return for default value Rmin = {Rmin}): ", Rmin)
Rmax = _get_float(f"Maximum horizon radius? (press return for default value Rmax = {Rmax}): ", Rmax)
Rnum = _get_int(f"Numer of R values? (press return for default value Rnum = {Rnum}): ", Rnum)
Rlinlog = _get_str(f"R value lin/log spacing? (press return for default = {Rlinlog}): ", Rlinlog)

amin = _get_float(f"Minimum a value? (press return for default value amin = {amin}): ", amin)
amax = _get_float(f"Maximum a value? (press return for default value amax = {amax}): ", amax)
anum = _get_int(f"Numer of a values? (press return for default value anum = {anum}): ", anum)

# time to run things, we'll do a for loops i guess
# i need to implement a way to save these arrays but idc rn

aArr = np.linspace(amin,amax,anum)

genAlphaPlot(Rmin,Rmax,Rnum,Rlinlog,norm=True)
genAlphaPlot(Rmin,Rmax,Rnum,Rlinlog,norm=False)

genShockPlot(Rmin,Rmax,Rnum,Rlinlog,norm=True)
genShockPlot(Rmin,Rmax,Rnum,Rlinlog,norm=False)

for a in aArr:
    genLvsrminPlot(Rmin,Rmax,Rnum,a,Rlinlog)

    genAdjLvsrminPlot(Rmin,Rmax,Rnum,a,Rlinlog)

    genUnAreaPlot(Rmin,Rmax,Rnum,a,Rlinlog,norm=True)
    genUnAreaPlot(Rmin,Rmax,Rnum,a,Rlinlog,norm=False)

    genMutInfPlot(Rmin,Rmax,Rnum,a,Rlinlog,norm=True)
    genMutInfPlot(Rmin,Rmax,Rnum,a,Rlinlog,norm=False)

    genLCritPlot(Rmin,Rmax,Rnum,a,Rlinlog,norm=True)
    genLCritPlot(Rmin,Rmax,Rnum,a,Rlinlog,norm=False)

# dict export
bigdict = {
    'fdate': fdate,
    'rmax': rmax,
    'num': num,
    'eps': eps,
    'Rmin': Rmin,
    'Rmax': Rmax,
    'Rnum': Rnum,
    'Rlinlog': Rlinlog,
    'amin': amin,
    'amax': amax,
    'anum': anum,
}

with open(f"{fpath}/{fnow}_data.json","w") as json_file:
    json.dump(bigdict,json_file,indent=4)

print(f'All plots and params saved to {fpath} and {fpathn}. \ngoodbye :3')
sys.exit(0)