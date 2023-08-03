"""
Combine all Treble .hdf5 data files in a directory into a single virtual hdf5 file.
This script can be run on original Treble .hdf5 files or virtual .vhdf5 files.
Virtual data files represent links to the original dataset, and must remain in the same relative location to be accessed.
"""

# INPUT PARAMETERS
#####################################################################################################################

# Define a directory containing multiple Treble .hdf5 files to combine.
# If no directory exists, run split_data_by_time.py on one of the files in ../sample_data/ to generate a directory of virtual .hdf5 files.
source_directory = "../sample_data/0.5s_split_files"

######################################################################################################################

from pathlib import Path
import h5py
from datetime import datetime
import numpy as np


def correct_timestamp_format(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('UTC-YMD%Y%m%d-HMS%H%M%S.%fZ')


def combine_hdf5_files(source_directory):
    source_files = list(Path(source_directory).glob("*.*hdf5"))
    output_file = Path(source_directory).joinpath("combined.h5")

    # Combine source data into single output file
    virtual_combine_datasets(source_files, output_file)

    # Copy all other attributes and data
    for dataset_name in [".", "data_product"]:
        copy_attributes(source_files[0], output_file, dataset_name)
    fix_timing_attributes(output_file)
    copy_group(source_files[0], output_file, "diagnostics")

    return output_file


def virtual_combine_datasets(source_files, output_filename):
    # Get dataset parameters from first file
    with h5py.File(source_files[0], "r") as f_in:
        dataset_names = list(f_in["data_product"].keys())
        nt = f_in["data_product/data"].shape[0]
        nx = f_in["data_product/data"].shape[1]

    with h5py.File(source_files[-1], "r") as f_in:
        nt_final = f_in["data_product/data"].shape[0]

    nt_total = nt*(len(source_files)-1) + nt_final

    # Create virtual layouts.
    data_layout = h5py.VirtualLayout(shape=(nt_total, nx), dtype=np.float32)
    posix_layout = h5py.VirtualLayout(shape=(nt_total,), dtype=np.float64)
    if "gps_time" in dataset_names:
        gps_layout = h5py.VirtualLayout(shape=(nt_total,), dtype=np.float64)

    # Add data from files into virtual layouts
    for n, fname in enumerate(source_files):
        with h5py.File(fname, "r") as f_in:
            data_layout[n*nt:(n+1)*nt, :] = h5py.VirtualSource(f_in["data_product/data"])
            posix_layout[n*nt:(n+1)*nt] = h5py.VirtualSource(f_in["data_product/posix_time"])
            if "gps_time" in dataset_names:
                gps_layout[n*nt:(n+1)*nt] = h5py.VirtualSource(f_in["data_product/gps_time"])

    # Create virtual datasets in output file
    with h5py.File(output_filename, "w") as f_out:
        f_out.create_virtual_dataset("data_product/data", data_layout, fillvalue=np.nan)
        f_out.create_virtual_dataset("data_product/posix_time", posix_layout, fillvalue=np.nan)
        if "gps_time" in dataset_names:
            f_out.create_virtual_dataset("data_product/gps_time", gps_layout, fillvalue=np.nan)

    # Copy attributes for copied datasets
    for dataset_name in dataset_names:
        copy_attributes(source_files[0], output_filename, f"data_product/{dataset_name}")


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


source_directory = Path(source_directory)
output_filename = combine_hdf5_files(source_directory)