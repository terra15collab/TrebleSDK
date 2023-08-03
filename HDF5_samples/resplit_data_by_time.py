"""
1. Combines all .hdf5 files in a directory into a single virtual combined.h5 file.
2. Split the combined file into multiple virtual .vhdf5 files of a specified duration.
Virtual datasets use relative path links to the original dataset, so must remain with the source data be accessed.
"""

# INPUT PARAMETERS
#####################################################################################################################

# Define a directory containing multiple Treble .hdf5 files to re-split.
# If no directory exists, run split_data_by_time.py on one of the files in ../sample_data/ to generate a directory of virtual .hdf5 files.
# source_directory = "../sample_data/0.5s_split_files"

# DESIRED OUTPUT FILE DURATION
file_duration = 0.5 # seconds

######################################################################################################################

import os
from pathlib import Path
import h5py
from datetime import datetime
import numpy as np


def correct_timestamp_format(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('UTC-YMD%Y%m%d-HMS%H%M%S.%fZ')


def split_hdf5_file(source_file, split_duration):
    print(f"Splitting {source_file} into {split_duration}s files.")
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
        print(f"Saving split data to to {dest_file}")

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
            relative=True
        )
        # splits time arrays
        for dset_name in ["posix_time", "gps_time"]:
            try:
                virtual_split_dataset(
                    f"data_product/{dset_name}",
                    source_file,
                    dest_file,
                    (n_split,),
                    i0,
                    relative=True
                )
            except KeyError:
                print(f"Dataset {dset_name} not found in original file.")

        # copies diagnostic data
        try:
            copy_group(source_file, dest_file, "diagnostics")
        except KeyError:
            print("No diagnostic data found in original file.")

        # copies attributes from original file
        copy_attributes(source_file, dest_file, ".")
        copy_attributes(source_file, dest_file, "data_product")
        fix_timing_attributes(dest_file)

        with h5py.File(dest_file, "r") as f_out:
            try:
                print(f"File Length: {f_out['data_product/gps_time'][-1] - f_out['data_product/gps_time'][0]}s")
            except KeyError:
                print(f"File Length: {f_out['data_product/posix_time'][-1] - f_out['data_product/posix_time'][0]}s")

    return save_directory, list(save_directory.glob("*.vhdf5"))


def virtual_split_dataset(dataset_name, source_fname, dest_fname, dest_shape, i0, relative=True):
    # gets source file parameters
    with h5py.File(source_fname, "r") as f_in:
        try:
            source_shape = f_in[dataset_name].shape
            dtype = f_in[dataset_name].dtype
        except KeyError:
            print(f"Dataset {dataset_name} not found in original file.")

    if relative:
        dp_path = os.path.relpath(source_fname, os.path.dirname(dest_fname))
    else: dp_path = source_fname
    with h5py.File(dest_fname, "a") as f_out:
        vsource = h5py.VirtualSource(dp_path, dataset_name, shape=source_shape, dtype=dtype)
        layout = h5py.VirtualLayout(shape=dest_shape, dtype=dtype)
        layout[:] = vsource[i0:i0 + dest_shape[0]]
        f_out.create_virtual_dataset(dataset_name, layout, fillvalue=0)

    # Duplicates attributes
    copy_attributes(source_fname, dest_fname, dataset_name)


def combine_hdf5_files(source_directory):
    source_directory = Path(source_directory)
    source_files = list(source_directory.glob("*.*hdf5"))
    source_files.sort()
    output_file = Path(source_directory).joinpath("combined.h5")

    print(f"Combining data in {source_directory}")

    # Combine source data into single output file
    virtual_combine_datasets(source_files, output_file)

    # Copy all other attributes and data
    for dataset_name in [".", "data_product"]:
        copy_attributes(source_files[0], output_file, dataset_name)
    fix_timing_attributes(output_file)
    try:
        copy_group(source_files[0], output_file, "diagnostics")
    except KeyError:
        print("No 'diagnostics' data found in source files.")

    print(f"Data saved to {output_file}")

    return output_file


def virtual_combine_datasets(source_files, output_filename, relative=True):
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

    # Add data from source files into virtual layouts
    for n, fname in enumerate(source_files):
        with h5py.File(fname, "r") as f_in:
            shape = f_in["data_product/data"].shape

        print(f"Adding data {shape} from {fname}")

        if relative:
            dp_path = Path(fname).relative_to(output_filename.parent)
        else: dp_path = fname

        data_layout[n*nt:(n+1)*nt, :] = h5py.VirtualSource(dp_path, "data_product/data", shape)
        posix_layout[n*nt:(n+1)*nt] = h5py.VirtualSource(dp_path, "data_product/posix_time", (shape[0],))
        if "gps_time" in dataset_names:
            gps_layout[n*nt:(n+1)*nt] = h5py.VirtualSource(dp_path, "data_product/gps_time", (shape[0],))

    # Create virtual datasets in output file
    with h5py.File(output_filename, "w") as f_out:
        f_out.create_virtual_dataset("data_product/data", data_layout)
        f_out.create_virtual_dataset("data_product/posix_time", posix_layout)
        if "gps_time" in dataset_names:
            f_out.create_virtual_dataset("data_product/gps_time", gps_layout)

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


source_directory = Path(source_directory).resolve()
combined_file = combine_hdf5_files(source_directory)
save_directory, split_files = split_hdf5_file(combined_file, file_duration)