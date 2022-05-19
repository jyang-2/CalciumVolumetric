import pprint as pp

import numpy as np


def in_range(x, x_start, x_end):
    """Return boolean vector where x is in [x_start, x_end)"""
    return np.logical_and(x >= x_start, x < x_end)


def temporal_downsample(ts, tsub, method='last'):
    """Downsample timestamps by factor tsub."""
    """
    Args:
        ts (np.ndarray): A 1D timestamp array (seconds)
        tsub (int): temporal downsampling factor
        method (str): 'first', 'last', 'mean'

    Returns:
        (np.ndarray) : downsampled timestamp vector
    """

    n_timestamps = len(range(tsub - 1, len(ts), tsub))

    if method == 'first':
        return ts[::tsub][:n_timestamps]
    elif method == 'last':
        return ts[tsub - 1::tsub]
    elif method == 'mean':
        ts_mean = (ts[::tsub][:n_timestamps] + ts[tsub - 1::tsub]) / 2
        return ts_mean


def split_and_downsample_timestamps(timestamps, tsub):
    split_stack_times_ds = [temporal_downsample(ts, 3) for ts in timestamps['split_stack_times']]

    split_timestamps = [None] * len(timestamps['split_stack_times'])
    for i, (ts, scope_ict, scope_fct) in enumerate(zip(split_stack_times_ds,
                                                       timestamps['scope_ict'],
                                                       timestamps['scope_fct'])):
        olf_mask = in_range(timestamps['olf_ict'], scope_ict, scope_fct)
        olf_ict = timestamps['olf_ict'][olf_mask]
        olf_fct = timestamps['olf_fct'][olf_mask]
        timestamps_split_ds = dict(stack_times=ts,
                                   scope_ict=scope_ict,
                                   scope_fct=scope_fct,
                                   olf_ict=olf_ict,
                                   olf_fct=olf_fct,
                                   n_stack_times=ts.size,
                                   n_olf_times=olf_ict.size,
                                   tsub=tsub
                                   )
        split_timestamps[i] = timestamps_split_ds
    return split_timestamps


def split_and_downsample_timestamps_file(ts_file, tsub):
    timestamps = np.load(ts_file, allow_pickle=True).item()
    pp.pprint(timestamps, compact=True)
    split_timestamps = split_and_downsample_timestamps(timestamps, tsub)
    for i, split_ts in enumerate(split_timestamps):
        save_file = f"timestamps__tsub_{tsub:02d}__acq_{i}.npy"
        save_file = ts_file.with_name(save_file)
        print(save_file)
        np.save(save_file, split_ts)
    return True
# %%


#
#
# # %%
# from pathlib import Path
#
# NAS_PROC_DIR = Path("/local/storage/Remy/narrow_odors/processed_data")
# fname_list = sorted(list(NAS_PROC_DIR.rglob("timestamps.npy")))
# for file in fname_list:
#     print('---')
#     print(file)
#
#     timestamps = np.load(file, allow_pickle=True).item()
#     tsub = 3
#     stack_times = temporal_downsample(timestamps['fixed_stack_times'], tsub)
#
#     timestamps_new = dict(frame_times=timestamps['frame_times'],
#                           stack_times=stack_times,
#                           olf_ict=timestamps['olf_ict'],
#                           olf_fct=timestamps['olf_fct'],
#                           scope_ict=timestamps['scope_ict'],
#                           scope_fct=timestamps['scope_ict'])
#     # pp.pprint(timestamps_new, compact=True)
#     np.save(file.with_name('timestamps.npy'), timestamps_new)
#     print(file.with_name('timestamps.npy'))
# # %%
# fname_ts = Path("/local/storage/Remy/natural_mixtures/processed_data/2022-02-10/1/kiwi/timestamps0.npy")
#
# timestamps = np.load(fname_ts, allow_pickle=True).item()
# tsub = 3
# stack_times = temporal_downsample(timestamps['fixed_stack_times'], tsub)
#
# timestamps_new = dict(frame_times=timestamps['frame_times'],
#                       stack_times=stack_times,
#                       olf_ict=timestamps['olf_ict'],
#                       olf_fct=timestamps['olf_fct'],
#                       scope_ict=timestamps['scope_ict'],
#                       scope_fct=timestamps['scope_ict'])
#
# np.save(timestamps_new, )


# %%

# # %%
#
# """
# Fixed timestamps.npy has fields:
# dict_keys(['frame_times', 'stack_times',
#  'scope_ict', 'scope_fct', 'olf_ict', 'olf_fct',
#  'split_frame_times', 'split_stack_times',
#  'n_frame_times_per_pulse', 'n_stack_times_per_pulse'])
# """
