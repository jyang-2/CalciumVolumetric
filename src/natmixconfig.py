from pathlib import Path

# prj = 'natural_mixtures'
prj = 'odor_space_collab'
# prj = 'odor_unpredictability'

if prj == 'natural_mixtures':
    NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/projects/natural_mixtures")
elif prj == 'odor_space_collab':
    NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/projects/odor_space_collab")
elif prj == 'odor_unpredictability':
    NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/projects/odor_unpredictability")

NAS_PROC_DIR = NAS_PRJ_DIR.joinpath("processed_data")
OLF_CONF_DIR = NAS_PRJ_DIR.joinpath("olfactometer_configs")
MANIFEST_FILE = NAS_PRJ_DIR.joinpath("manifestos", "flat_linked_thor_acquisitions.json")


mov_dir_files = ['stim_list.json',
                 'timestamps.npy',
                 'pid.npz',
                 'xrda_pid_traces.nc',
                 'xrda_pid_trials.nc']
