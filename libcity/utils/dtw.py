import numpy as np
try:
    from dtaidistance import dtw, dtw_ndim
except Exception:
    dtw = None
    dtw_ndim = None


def dtw_distance(x, y, radius=None):
    x_arr = np.asarray(x, dtype=np.double)
    y_arr = np.asarray(y, dtype=np.double)
    window = None if radius is None else int(radius)
    if dtw is None:
        raise RuntimeError('dtaidistance package is required for dtw_distance')
    return dtw.distance_fast(x_arr, y_arr, window=window, use_pruning=True)


def dtw_distance_ndim(x, y, radius=None):
    x_arr = np.asarray(x, dtype=np.double)
    y_arr = np.asarray(y, dtype=np.double)
    window = None if radius is None else int(radius)
    if dtw_ndim is None:
        raise RuntimeError('dtaidistance package is required for dtw_distance_ndim')
    return dtw_ndim.distance_fast(x_arr, y_arr, window=window, use_pruning=True)
