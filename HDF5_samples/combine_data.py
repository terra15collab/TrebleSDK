"""
Combine all Treble .hdf5 data files in a directory into a single virtual hdf5 file.
This script can be run on a directory containing only virtual .hdf5 files.
Virtual data files represent links to the original dataset, and must remain with the original data to be accessed.
"""

# INPUT PARAMETERS
#####################################################################################################################

# Define a directory containing multiple Treble .hdf5 files to combine.
source_directory = "../sample_data/0.5s_split_files"

######################################################################################################################

from pathlib import Path

import h5py
from datetime import datetime
import numpy as np


def correct_timestamp_format(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('UTC-YMD%Y%m%d-HMS%H%M%S.%fZ')


def combine_hdf5_files(source_directory):
    files = list(Path(source_directory).glob("*.*hdf5"))

    # check parameters of datasets in original files
    with h5py.File(files[0], "r") as f_in:
        dataset_names = list(f_in["data_product"].keys())
        nt = f_in["data_product"]["data"].shape[0]
        nx = f_in["data_product"]["data"].shape[1]

    # check length of last file
    with h5py.File(files[-1], "r") as f_in:
        nt_final = f_in["data_product"]["data"].shape[0]

    nt_total = nt*(len(files)-1) + nt_final

    # create Virtual Layouts.
    data_layout = h5py.VirtualLayout(shape=(nt_total, nx), dtype=np.float32)
    posix_layout = h5py.VirtualLayout(shape=(nt_total,), dtype=np.float64)
    if "gps_time" in dataset_names:
        gps_layout = h5py.VirtualLayout(shape=(nt_total,), dtype=np.float64)

    # add virtual source from each file to virtual layouts
    for n, fname in enumerate(files):
        with h5py.File(fname, "r") as f_in:
            data_group = f_in["data_product"]
            dataset_names = list(data_group.keys())

            # add data to virtual layout
            data_layout[n*nt:(n+1)*nt, :] = h5py.VirtualSource(data_group["data"])

            # add posix time to virtual layout
            posix_layout[n*nt:(n+1)*nt] = h5py.VirtualSource(data_group["posix_time"])

            # add gps time to virtual layout
            if "gps_time" in dataset_names:
                gps_layout[n*nt:(n+1)*nt] = h5py.VirtualSource(data_group["gps_time"])

    # define output file
    output_filename = Path(source_directory).joinpath("combined.h5")
    # add virtual datasets to output file
    with h5py.File(output_filename, "w") as f_out:
        # create virtual dataset
        f_out.create_virtual_dataset("data_product/data", data_layout, fillvalue=np.nan)
        f_out.create_virtual_dataset("data_product/posix_time", posix_layout, fillvalue=np.nan)
        if "gps_time" in dataset_names:
            f_out.create_virtual_dataset("data_product/gps_time", gps_layout, fillvalue=np.nan)

        # copy attributes from first file
        with h5py.File(files[0], "r") as f_in:
            # copy top-level attributes
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value

            # copy group-level attributes
            for key, value in f_in["data_product"].attrs.items():
                f_out["data_product"].attrs[key] = value

    return output_filename


source_directory = Path(source_directory)
output_filename = combine_hdf5_files(source_directory)