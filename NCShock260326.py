# interactive script to plot non-com shocked entropies

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