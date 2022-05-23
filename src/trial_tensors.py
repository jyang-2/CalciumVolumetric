import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


def split_traces_to_trials(cells_x_time, ts, stim_ict, trial_ts):
    """
    Extracts periods of activity defined by `trial_ts` centered at `stim_ict`.

    Extracted trials should be the same length.

    Args:
        cells_x_time (np.ndarray):
        ts (np.ndarray):
        stim_ict (Union[List, np.ndarray]):
        trial_ts (np.ndarray): timestamps for trial interval, with stim_ict corresponding to 0

    Returns:
        (List[np.ndarray]): cell x trial subarrays, split according to trial_ts.

    Examples:
        >>>> trial_timestamps = np.arange(-5, 20, 0.5)
        >>>> trial_traces = split_traces_to_trials(Fc, ts, olf_ict, trial_timestamps)
        >>>> ttrials =  np.stack(trial_traces, axis=0)
    """

    interp_traces = interp1d(ts, cells_x_time, axis=-1)

    F_trials = []

    for ict in stim_ict:
        F_trials.append(interp_traces(trial_ts + ict))

    return F_trials


def make_trial_tensor(cells_x_time, ts, stim_ict, trial_ts):
    """
    Converts neuron activity array into a 3D tensor, w/ dimensions trials x neurons x time.

    Trial intervals are defined by ts, stim_ict, and trial_ts.

    If you want the time period from 5 sec. before stimulus onset to 20 seconds after stimulus onset,
    trial_ts should be something like np.arange(-5, 20, 0.2).

    This will also result in the data being interpolated to a frame rate of 1/dt, or 1/0.2 = 5


    Args:
        cells_x_time (np.ndarray):
        ts (np.ndarray):
        stim_ict (Union[List, np.ndarray]):
        trial_ts (np.ndarray): timestamps for trial interval, with stim_ict corresponding to 0

    Returns:
        (np.ndarray): cell x trial subarrays, split according to trial_ts.

    Examples:
        >>>> trial_timestamps = np.arange(-5, 20, 0.5)
        >>>> trial_tensors = make_trial_tensors(Fc, ts, olf_ict, trial_timestamps)
        >>>> ttrials.shape
                (57, 1847, 1690)
    """
    F_trials = split_traces_to_trials(cells_x_time, ts, stim_ict, trial_ts)
    trial_tensor = np.stack(F_trials, axis=0)
    return trial_tensor


def make_xrds_trial_tensor(ds, trial_ts=None):
    """
    Takes xarray dataset w/ dims (cells, time) and converts into (trials, cells, time).

    Args:
        ds (xr.Dataset): has dims (cells, time)
        trial_ts (np.ndarray): 1D time vector, for time interval around stimulus onset
                                (t=0 is when the stimulus turns on)

    Returns:
        xr.Dataset: has dims (trials, cells, time)

    """
    if trial_ts is None:
        trial_ts = np.arange(-5, 20, 0.5)

    attrs = ds.attrs.copy()

    dim_names = list(ds.coords.dims)
    coord_names = list(ds.coords.keys())
    nondim_coords = [item for item in coord_names if item not in dim_names]

    # create values for new dataset dimensions
    n_trials = len(ds.attrs['stim'])
    cell_var = list(ds.dims)[0]
    cells = ds.coords[cell_var].to_numpy()

    data_vars = {}
    for k, v in ds.data_vars.items():
        trials_x_cells_x_time = make_trial_tensor(v.to_numpy(),
                                                  ts=ds.time.to_numpy(),
                                                  stim_ict=ds.attrs['olf_ict'],
                                                  trial_ts=trial_ts)
        data_vars[k] = (("trials", cell_var, "time"), trials_x_cells_x_time)

    ds_trial_tensors = xr.Dataset(
        data_vars=data_vars,
        coords={'trials': range(n_trials),
                cell_var: cells,
                'time': trial_ts,
                'stim': ('trials', ds.attrs['stim'])
                },
        attrs=attrs
    )
    # copy non-time dimension coords over
    for cname in nondim_coords:
        if 'time' not in ds.coords[cname].dims:
            ds_trial_tensors = ds_trial_tensors.assign_coords({cname: (ds.coords[cname].dims, ds.coords[cname].values)})

    return ds_trial_tensors
