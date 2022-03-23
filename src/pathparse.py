# import plotly.express as px
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import seaborn as sns

NAS_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")

stat_file_list = sorted(list(NAS_PROC_DIR.rglob('downsampled_3/**/combined/stat.npy')))


def get_mongodb_query(filepath):
    """ Creates mongodb query w/ fields date_imaged, fly_num, and thorimage_name.

    Examples::

        dquery = pathparse.get_mongodb_query("/local/storage/Remy/natural_mixtures/
                    processed_data/2022-02-11/1/kiwi_ea_eb_only/downsampled_3")

        lacq = db[ 'linked_thor_acq_collection'].find_one(dquery)

    """
    date_imaged, fly_num, thorimage_name = get_mov_dir(filepath).parts[-3:]
    return dict(date_imaged=date_imaged, fly_num=int(fly_num), thorimage_name=thorimage_name)


def is_date_matching(date_str):
    """Checks if string can be parsed to date.

    Date must be formatted like '2022-02-11'.
    """
    try:
        return bool(datetime.strptime(date_str, '%Y-%m-%d'))
    except ValueError:
        return False


def get_date_from_path(filepath):
    """
    If any of the file parts can be parsed to a date, returns date as str.
    Otherwise, returns None.

    Args:
        filepath (Path): filepath
    """
    if isinstance(filepath, str):
        fileparts = Path(filepath).parts
    elif isinstance(filepath, Path):
        fileparts = filepath.parts

    for item in fileparts:
        if is_date_matching(item):
            return item
    return None


def get_mov_dir(stat_file):
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)
    return NAS_PROC_DIR.joinpath(*stat_file.relative_to(NAS_PROC_DIR).parts[:3])


def load_trial_timing(stat_file):
    timestamps = np.load(NAS_PROC_DIR.joinpath(get_timestamps_rel_path(stat_file)), allow_pickle=True).item()
    return timestamps


def get_downsampling_dir(filepath):
    """
    Returns path to folder w/ downsampled data. Folder should be named something like 'downsampled_3'.

    If there is no downsampling directory in the filepath, returns None.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    for folder in filepath.parents:
        if 'downsampled_' in folder.name:
            return folder
    return None


def tsub_from_path(filepath):
    """Extracts temporal downsampling factor from filepath."""

    dsub_folder = get_downsampling_dir(filepath)
    if dsub_folder is not None:
        return temporal_downsampling_factor(dsub_folder.name)
    else:
        return 1


def temporal_downsampling_factor(folder_name):
    """ Parse folder name (ex: 'downsampled_3') for downsampling factor.

    Temporal downsampling factor has either no prefix ('downsampled_3') or '_tsub' prefix ('downsampled_tsub3')

    Args:
        folder_name (str): directory name (contains substring 'downsampled')

    Returns:
        tsub (int): temporal downsampling factor, default tsub=1 if no downsampling
    """

    if 'tsub' in folder_name:
        r = parse.search("_tsub{tsub:d}", folder_name)
    else:
        r = parse.search("_{tsub:d}", folder_name)

    if r is None:
        tsub = 1
    else:
        tsub = r['tsub']

    return tsub


def get_timestamps_rel_path(stat_file):
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)
    return Path(*stat_file.relative_to(NAS_PROC_DIR).parts[:3], 'timestamps.npy')


# %%
def plot_total_F(stat_file):
    """ Sums fluorescence of all extracted components and plots it

    Args:
        stat_file: Path to suite2p 'stat.npy' file
    """

    F = np.load(stat_file.with_name('F.npy'))
    Fneu = np.load(stat_file.with_name('Fneu.npy'))
    iscell, cellprob = np.load(stat_file.with_name('iscell.npy')).T

    Fc = F - 0.7 * Fneu

    fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(11, 8.5), tight_layout=True)

    ax1 = axarr[0]
    ax1.plot(F.sum(axis=0), label='F')
    ax1.plot(Fneu.sum(axis=0), label='Fneu')
    ax1.set_title('total fluorescence (suite2p)')
    ax1.legend()

    ax2 = axarr[1]
    ax2.plot(Fc.mean(axis=0), label='cellprob>0')

    cls_thresh = np.arange(0, 0.6, 0.1)
    # ax2.set_prop_cycle('color', plt.cm.Spectral(len(cls_thresh)))
    for thr in cls_thresh:
        ax2.plot(Fc[cellprob > thr, :].sum(axis=0),
                 label=f"cellprob>{thr:0.1f}",
                 )
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.95))
    ax2.set_title('Fc = F - 0.7*Fneu')

    fig.suptitle(stat_file.relative_to(NAS_PROC_DIR))
    return fig, axarr


# %%
def plot_pca_results(stat_file, do_kde=True):
    df = pd.read_csv(stat_file.with_name('df_pca.csv'), index_col=0)
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(11, 8.5), tight_layout=True)
    labels = [format_stim_str(item) for item in df['stim'].tolist()]

    df['labels'] = labels
    sns.kdeplot(data=df, x="x", y="y", hue="labels", ax=ax, hue_order=np.unique(sorted(labels)).tolist(),
                # fill=True, alpha=0.3,
                # thresh=0.1
                )
    sns.scatterplot(data=df, x="x", y="y", hue="labels", ax=ax, hue_order=np.unique(sorted(labels)).tolist())

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.set_title(stat_file.relative_to(NAS_PROC_DIR))
    return fig, ax


def format_stim_str(stim):
    s = stim.replace('ethanol', 'EtOH')
    s = s.replace('isoamyl acetate', 'IaA')
    s = s.replace('isoamyl acetate', 'IaAA')
    s = s.replace('kiwi approx.', 'kiwi')
    s = s.replace('ethyl butyrate', 'eb')
    s = s.replace('ethyl acetate', 'ea')
    s = ", ".join(sorted(s.split(", ")))
    return ", ".join(sorted(s.split(", ")))
# %%
# labels = [format_stim_str(item) for item in df['stim']]
#
# fig = px.scatter(df, x="x", y="y", text=labels, color=labels)
# fig.update_traces(textposition='bottom right')
# fig.update_layout(
#     height=600,
#     width=600,
#     title_text='PCA for peak_amp, computed from C_dec'
# )
# fig.show()


# #%%
# #NAS_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")
# csv_list = sorted(list(NAS_PROC_DIR.rglob('df_pca.csv')))
#
# fig_list = []
# for file in csv_list:
#     fig, ax = plot_pca_results(file)
#     fig_list.append(fig)
#     plt.show()
#
# with PdfPages(NAS_PROC_DIR.parent.joinpath('PCA_peak_amp_from_C_dec_okcmap.pdf')) as pdf:
#     for fig in fig_list:
#         pdf.savefig(fig)
#
#
# #%% compute mean responses for cluster identities
# # plot grouped heatmaps
# stat = np.load(stat_file, allow_pickle=True)
#
# # convert scalar stat fields into a DataFrame
# stat_scalar = [{k: v for k, v in item.items() if np.isscalar(v)} for item in stat]
# df_stat = pd.DataFrame.from_dict(stat_scalar, orient='columns')
#
# F = np.load(stat_file.with_name('F.npy'))
# Fneu = np.load(stat_file.with_name('Fneu.npy'))
# iscell, cellprob = np.load(stat_file.with_name('iscell.npy')).T
# iscell = iscell==1
# Fc = F - 0.7 * Fneu
# timestamps = load_trial_timing(stat_file)
# ts = timestamps['fixed_stack_times'][2::3]
# #%%
# fig, axarr = plt.subplots(2, 2, figsize=(11, 8.5), tight_layout=True,
#                           sharex='col', gridspec_kw={'height_ratios': [1, 5], 'width_ratios': [10, 1]})
# model = np.load(stat_file.with_name('embedding.npy'), allow_pickle=True).item()
# if isinstance(model, dict):
#     cluster_traces = np.zeros((model['n_X'], Fc.shape[1]))
#     for gid in np.unique(model['xid']):
#         cluster_traces[gid, :] = Fc[model['xid'] == gid, :].mean(axis=0)
#     sns.heatmap(cluster_traces, square=False, yticklabels=5, xticklabels=100,
#                 # cbar_ax=axarr[0, 1],
#                 cbar_kws=dict(shrink=0.25),
#                 ax=axarr[1, 0])
#
# plt.show()
#
