# interactive script to plot non-com shocked entropies

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

date = datetime.date.today()
now = datetime.datetime.now()
fdate = date.strftime('%y%m%d')
fnow = now.strftime('%y%m%d_%H%M%S')

fpath = f'./NCOut_{fdate}'
os.makedirs(fpath,exist_ok=True)

print('Entropy calculations for non-commutative shocked AdS black hole.')

