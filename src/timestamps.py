import numpy as np


def temporal_downsample(ts, tsub, method='last'):
    """
    Args:
        ts (np.ndarray): A 1D timestamp array (seconds)
        tsub (int): temporal downsampling factor
        method (str): 'first', 'last', 'mean'

    Returns:
        (np.ndarray) : downsampled timestamp vector
    """

    n_timestamps = len(range(tsub-1, len(ts), tsub))

    if method == 'first':
        return ts[::tsub][:n_timestamps]
    elif method == 'last':
        return ts[tsub - 1::tsub]
    elif method == 'mean':
        ts_mean = (ts[::tsub][:n_timestamps] + ts[tsub - 1::tsub])/2
        return ts_mean



