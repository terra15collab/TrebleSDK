"""
Split a single Treble data file into multiple virtual hdf5 files (.vhdf5) of a specified time duration.
Virtual data files represent links to the original dataset, and must remain in the same relative directory to be accessed.
"""

# INPUT PARAMETERS
#####################################################################################################################

# Define the path to the .h5 virtual hdf5 file or single .hdf5 file to split.
# hdf5_data_file = "../sample_data/v5_compatible.h5"
# OR
hdf5_data_file = "../sample_data/example_triggered_shot.hdf5"

# DESIRED OUTPUT FILE DURATION
file_duration = 0.5 # seconds

######################################################################################################################

from pathlib import Path
import h5py
from datetime import datetime
import numpy as np


def correct_timestamp_format(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('UTC-YMD%Y%m%d-HMS%H%M%S.%fZ')


def split_hdf5_file(source_file, split_duration):
    save_directory = source_file.parent.joinpath(f"{split_duration:.1f}s_split_files")
    save_directory.mkdir(exist_ok=True)

    # Get split parameters from source file
    with h5py.File(source_file, "r") as f_in:
        # determines number of samples in each split file
        n_split = int(split_duration/f_in.attrs.get("dt_computer"))
        nt = f_in["data_product/data"].shape[0]
        nx = f_in["data_product/data"].shape[1]

        # determines split indexes and times from original dataset
        split_indices = np.arange(0, nt, n_split)
        try:
            split_times = f_in["data_product/gps_time"][split_indices]
        except KeyError:
            split_times = f_in["data_product/posix_time"][split_indices]

    # Creates mapping to new .vhdf5 files
    for n, (i0, t0) in enumerate(zip(split_indices[:], split_times[:])):
        dest_file = save_directory.joinpath(f"{correct_timestamp_format(t0)}_seq_{str(n).zfill(11)}.vhdf5")
        print(dest_file)

        # checks for last file case.
        if i0 + n_split > nt:
            n_split = nt - i0

        # splits main dataset
        virtual_split_dataset(
            "data_product/data",
            source_file,
            dest_file,
            (n_split, nx),
            i0,
        )
        # splits time arrays
        for dset_name in ["posix_time", "gps_time"]:
            try:
                virtual_split_dataset(
                    f"data_product/{dset_name}",
                    source_file,
                    dest_file,
                    (n_split,),
                    i0
                )
            except KeyError:
                print(f"Dataset {dset_name} not found in original file.")

        # copies diagnostic data
        copy_group(source_file, dest_file, "diagnostics")

        # copies attributes from original file
        copy_attributes(source_file, dest_file, ".")
        copy_attributes(source_file, dest_file, "data_product")
        fix_timing_attributes(dest_file)

        with h5py.File(dest_file, "r") as f_out:
            print(f_out["data_product/data"])
            try:
                print(f"File Length: {f_out['data_product/gps_time'][-1] - f_out['data_product/gps_time'][0]}s")
            except KeyError:
                print(f"File Length: {f_out['data_product/posix_time'][-1] - f_out['data_product/posix_time'][0]}s")

    return save_directory, list(save_directory.glob("*.vhdf5"))


def virtual_split_dataset(dataset_name, source_fname, dest_fname, dest_shape, i0):
    # gets source file parameters
    with h5py.File(source_fname, "r") as f_in:
        try:
            source_shape = f_in[dataset_name].shape
            dtype = f_in[dataset_name].dtype
        except KeyError:
            print(f"Dataset {dataset_name} not found in original file.")

    # Maps source file to virtual dataset
    with h5py.File(dest_fname, "a") as f_out:
        vsource = h5py.VirtualSource(source_fname, dataset_name, shape=source_shape, dtype=dtype)
        layout = h5py.VirtualLayout(shape=dest_shape, dtype=dtype)
        layout[:] = vsource[i0:i0 + dest_shape[0]]
        f_out.create_virtual_dataset(dataset_name, layout, fillvalue=0)

    # Duplicates attributes
    copy_attributes(source_fname, dest_fname, dataset_name)


def copy_attributes(source_file, dest_file, group_name):
    with h5py.File(source_file, "r") as f_in:
        with h5py.File(dest_file, "a") as f_out:
            attrs = dict(f_in[group_name].attrs)
            f_out[group_name].attrs.update(attrs)


def copy_group(source_file, dest_file, dataset_name):
    with h5py.File(source_file, "r") as f_in:
        with h5py.File(dest_file, "a") as f_out:
            f_in.copy(dataset_name, f_out)


def fix_timing_attributes(dest_file):
    with h5py.File(dest_file, "a") as f_out:
        correct_attributes = {
            "file_start_computer_time": f_out["data_product/posix_time"][0],
            "file_start_computer_time_string": correct_timestamp_format(f_out["data_product/posix_time"][0]),
            "nt": f_out["data_product/data"].shape[0],
        }
        try: correct_attributes.update({"file_start_gps_time": f_out["data_product/gps_time"][0]})
        except KeyError: correct_attributes.update({"file_start_gps_time":f_out["data_product/posix_time"][0]})

        f_out.attrs.update(correct_attributes)


hdf5_data_file = Path(hdf5_data_file).resolve()
save_directory, split_files = split_hdf5_file(hdf5_data_file, file_duration)