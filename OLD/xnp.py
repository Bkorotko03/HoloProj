#conditional import for cupy or numpy/scipy
#i dont feel like rewriting this a million times...

try:
    import cupy as cp
    try:
        has_cuda = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        has_cuda = False
except Exception:
    cp = None
    has_cuda = False

if has_cuda:
    np = cp
    import cupyx.scipy as sp
    print("Using CuPy on an Nvidia GPU")
else:
    import numpy as np
    import scipy as sp
    print("Using NumPy and SciPy on CPU")