"""
Script to split a single Treble data file into multiple virtual hdf5 files (.vhdf5) of a specified time duration.
Virtual data files represent links to the original dataset, and must remain in the same directory to be accessed.
"""

# INPUT PARAMETERS
#####################################################################################################################

# Define the path to the .h5 virtual hdf5 file or single .hdf5 file to split.
hdf5_data_file = "../sample_data/v5_compatible.h5"
# OR
# hdf5_data_file = "../sample_data/example_triggered_shot.hdf5"

# DESIRED OUTPUT FILE DURATION
file_duration = 0.5 # seconds
######################################################################################################################

from pathlib import Path

import h5py
from datetime import datetime
import numpy as np

def correct_timestamp_format(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('UTC-YMD%Y%m%d-HMS%H%M%S.%fZ')

def split_hdf5_file(source_hdf5_file, split_duration):
    save_directory = hdf5_data_file.parent.joinpath(f"{split_duration:.1f}s_split_files")
    save_directory.mkdir(exist_ok=True)

    # Get original file parameters
    with h5py.File(source_hdf5_file, "r") as f_in:
        print("Splitting Datasets from Original File:")
        print(source_hdf5_file)

        metadata = dict(f_in.attrs)
        group = f_in["data_product"]
        dataset_names = list(group.keys())

        dataset_info = {}
        for dset_name in dataset_names:
            print(group[dset_name])
            dataset_info[dset_name] = dict(group[dset_name].attrs)

        # determines number of samples in each split file
        n_split = int(split_duration/metadata['dt_computer'])
        nt = group['data'].shape[0]
        nx = group['data'].shape[1]

        # determines split indexes and times from original dataset
        split_indexes = np.arange(0, nt, n_split)
        if "gps_time" in dataset_names:
            split_times = group["gps_time"][split_indexes]
        else:
            split_times = group["posix_time"][split_indexes]

    # Creates virtual mapping to new .vhdf5 files
    split_filenames = []
    for n, (i1, t1) in enumerate(zip(split_indexes[:], split_times[:])):
        dest_fname = save_directory.joinpath(f"{correct_timestamp_format(t1)}.vhdf5")
        split_filenames.append(dest_fname)
        print(dest_fname)

        with h5py.File(dest_fname, "w") as f_out:
            # checks for last file case.
            if i1 + n_split > nt:
                n_split = nt - i1
            # creates virtual dataset for "data" dataset
            layout = h5py.VirtualLayout(shape=(n_split, nx), dtype=np.float32)
            vsource = h5py.VirtualSource(source_hdf5_file, "data_product/data", shape=(nt, nx), dtype=np.float32)
            layout[:] = vsource[i1:i1 + n_split]
            f_out.create_virtual_dataset("data_product/data", layout, fillvalue=0)
            print(f_out['data_product/data'])

            # creates virtual datasets for time arrays
            for dset_name in dataset_names[1:]:
                layout = h5py.VirtualLayout(shape=(n_split,), dtype=np.float64)
                vsource = h5py.VirtualSource(source_hdf5_file, f"data_product/{dset_name}", shape=(nt, ), dtype=np.float64)
                layout[:] = vsource[i1:i1 + n_split]
                f_out.create_virtual_dataset(f"data_product/{dset_name}", layout, fillvalue=0)
                print(f_out[f"data_product/{dset_name}"])

            # change key top-level attributes
            metadata["file_start_computer_time"] = f_out["data_product/posix_time"][0]
            metadata["file_start_computer_time_string"] = correct_timestamp_format(f_out["data_product/posix_time"][0])
            if "gps_time" in dataset_names:
                metadata["file_start_gps_time"] = f_out["data_product/gps_time"][0]
                print(f"File Duration: {f_out['data_product/gps_time'][-1] - f_out['data_product/gps_time'][0]}s")
            else:
                metadata["file_start_gps_time"] = "FALSE"
                print(f"File Duration: {f_out['data_product/posix_time'][-1] - f_out['data_product/posix_time'][0]}s")

            metadata["nt"]=n_split
            f_out.attrs.update(metadata)

            # Copies attributes for each dataset
            for dset_name, attrs in dataset_info.items():
                f_out[f"data_product/{dset_name}"].attrs.update(attrs)

    return split_filenames


hdf5_data_file = Path(hdf5_data_file)
split_filenames = split_hdf5_file(hdf5_data_file, file_duration)