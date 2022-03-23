import xarray as xr


def make_xrds_traces(traces, cell_ids, ts, attrs=None):
    data_vars = {k: (["cells", "time"], traces[k]) for k, v in traces.items()}
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            cells=cell_ids,
            time=ts
        ),
        attrs=attrs
    )
    return ds


def make_xrda_traces(trials_x_cells_x_time, stim_list, cell_ids, ts, attrs=None):
    n_trials, n_cells, n_timepoints = trials_x_cells_x_time.shape

    da = xr.DataArray(
        data=trials_x_cells_x_time,
        dims=['trials', 'cells', 'time'],
        coords=dict(
            trials=range(n_trials),
            cells=cell_ids,
            time=ts,
            stim=('trials', stim_list)
        ),
        attrs=attrs
    )
    return da

# def make_xrda_clustmeans():
#     return None


