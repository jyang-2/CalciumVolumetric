from typing import List

import pydantic

import natmixconfig
import olf_conf
from aggregate import PanelMovies, filter_flacq_list
from pydantic_models import FlatFlyAcquisitions
# from scripts import step05_pool_activation_strengths
from scripts import step00_suite2p_add_extras, \
    step01_suite2p_to_xarray, \
    step02_suite2p_make_suite2p_outputs_xid0, \
    step04_suite2p_traces_and_trials, \
    step05b_compute_trial_peaks, \
    step05c_compute_trial_rdms, \
    convert_suite2p_reg_to_xarray

# from PyPDF2 import PdfFileMerger

# set data directories, metadata filepath
# PRJ_DIR = Path("/local/matrix/Remy-Data/projects/odor_space_collab")
# PRJ_DIR = Path("/local/matrix/Remy-Data/projects/natural_mixtures")
# %%
prj = natmixconfig.NAS_PRJ_DIR.name

all_flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], natmixconfig.MANIFEST_FILE)
# %%
# load flat fly acquisitions (metadata)
if 'natural_mixtures' in prj:
    flat_acqs = list(filter(lambda x: x.stat_file() is not None, flat_acqs))
    flat_acqs = list(filter(lambda x: not x.is_pair(), flat_acqs))
elif 'odor_space_collab' in prj:
    flat_acqs = list(filter(lambda x: x.stat_file() is not None, flat_acqs))
    flat_acqs = list(filter(lambda x: x.movie_type == 'megamat0', flat_acqs))

    # flat_acqs = [flat_acqs[1]]
    # flat_acqs = list(filter(lambda x: x.movie_type in ['validation0', 'validation1'], flat_acqs))
elif 'odor_unpredictability' in prj:
    pass
# %%
odorspace_panel = PanelMovies(prj=prj, panel='kiwi')

odorspace_flat_acqs = filter_flacq_list(
        flat_acqs,
        allowed_imaging_type='kc_soma',
        allowed_movie_types=odorspace_panel.movies,
        has_s2p_output=True)

for item in odorspace_flat_acqs:
    print(item.filename_base())
    print('\n')
# %%
megamat0_panel = PanelMovies(prj=prj, panel='megamat')

flat_acqs = filter_flacq_list(
        all_flat_acqs,
        allowed_imaging_type='kc_soma',
        allowed_movie_types=megamat0_panel.movies,
        has_s2p_output=True)

print('\n')
print('megamat0 flat_acqs:')
print('-------------------')
for item in flat_acqs:
    print(item.filename_base())
# %%
megamat_panel = PanelMovies(prj=prj, panel='kiwi')

flat_acqs = filter_flacq_list(
        all_flat_acqs,
        allowed_imaging_type=None,
        allowed_movie_types=megamat_panel.movies,
        has_s2p_output=True)

print('\n')
print('megamat flat_acqs:')
print('-------------------')
for item in flat_acqs:
    print(item.filename_base())

# %% parse olf_config yaml to pin_odor_list.json, stim_list.json, etc.

# flat_acqs = flat_acqs[-1:]
run_post_suite2p_steps = True
run_post_xid0_steps = True

if __name__ == '__main__':
    OLF_CONFIG_DIR = natmixconfig.NAS_PRJ_DIR.joinpath("olfactometer_configs")

    for flacq in flat_acqs[:3]:
        mov_dir = flacq.mov_dir(relative_to=natmixconfig.NAS_PROC_DIR)

        # parse olf_config yaml (make stimulus .json files)
        if not mov_dir.joinpath('stim_list.json').is_file():
            olf_data = olf_conf.main(OLF_CONFIG_DIR.joinpath(flacq.olf_config), mov_dir)

        ################################################
        # for FlatFlyAcquisitions with suite2p outputs
        ################################################
        if run_post_suite2p_steps:
            if flacq.stat_file() is not None:
                print(flacq.stat_file)
                stat_file = natmixconfig.NAS_PROC_DIR / flacq.stat_file()

                # add extra outputs to suite2p results
                if not stat_file.with_name('cell_stats').is_dir():
                    step00_suite2p_add_extras.main(stat_file)

                # convert suite2p outputs to xarray
                if not stat_file.with_name('xrds_suite2p_outputs.nc').is_file():
                    step01_suite2p_to_xarray.main(flacq)

                if run_post_xid0_steps:
                    # add good_xids (xid0) to xarray dataset
                    run_step02 = not stat_file.with_name('xrds_suite2p_outputs_xid0.nc').is_file()
                    # run_step02 = False
                    if run_step02:
                        step02_suite2p_make_suite2p_outputs_xid0.main(flacq)

                    # Maybe figure out what signal smoothing/filtering would be best here?
                    # if not stat_file.with_name('xrds_suite2p_traces_xid0.nc').is_file():
                    #     step03_suite2p_process_traces.main(flacq)

                    # convert to trial structured dataset (requires xrds_suite2p_outputs_xid0.nc)
                    if not stat_file.with_name('xrds_suite2p_trials_xid0.nc').is_file():
                        step04_suite2p_traces_and_trials.main(flacq)

                    # make trial respvec peaks
                    # run_step05b = False
                    run_step05b = not all(step05b_compute_trial_peaks.has_output_files(stat_file))
                    if run_step05b:
                        if flacq.imaging_type == 'kc_soma' or flacq.imaging_type is None:
                            peak_range = (2, 8)
                            baseline_range = (-5, 0)
                        if flacq.imaging_type == 'pn_boutons':
                            peak_range = (0, 3)
                            baseline_range = (-5, -0.5)
                        (ds_mean_peak,
                         ds_max_peak,
                         ds_maxidx) = step05b_compute_trial_peaks.main(flacq,
                                                                       save_files=True,
                                                                       peak_range=peak_range,
                                                                       baseline_range=baseline_range
                                                                       )

                    # if not stat_file.with_name('RDM_trials').is_dir():
                    # step05c_compute_trial_rdms.main(flacq, save_files=True)
                    run_step05c = True
                    if run_step05c:
                        step05c_compute_trial_rdms.main(flacq, metrics=['correlation', 'cosine'],
                                                        save_files=True, make_plots=True)

                    # compute dff images
                    # da_reg = convert_suite2p_reg_to_xarray\
                    #     .load_registered_suite2p_movie_as_xarray_from_flacq(flacq)
                    run_dff_images = True
                    if run_dff_images:
                        if flacq.imaging_type == 'kc_soma':
                            peak_win = (2, 8)
                            baseline_win = (-10, -0.05)
                            tiff_dir = convert_suite2p_reg_to_xarray.main(
                                    flacq,
                                    baseline_win=baseline_win,
                                    peak_win=peak_win,
                                    sigma_baseline=1,
                                    sigma_peak=0.5,
                                    smooth_baseline=True,
                                    smooth_peak=False,
                                    dff_method='dff',
                                    save_netcdf=True)
                        if flacq.imaging_type == 'pn_boutons':
                            peak_win = (.25, 2)
                            baseline_win = (-5, -0.5)
                            tiff_dir = convert_suite2p_reg_to_xarray.main(
                                    flacq,
                                    baseline_win=baseline_win,
                                    peak_win=peak_win,
                                    sigma_baseline=0.5,
                                    sigma_peak=0.5,
                                    smooth_baseline=True,
                                    smooth_peak=True,
                                    dff_method='dff',
                                    save_netcdf=True)



# %%
