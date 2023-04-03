"""
Script to crop out multiple triggered sections of data from a continuous Treble dataset based on a list of trigger times.
Trigger times are defined as strings in a .csv file.

To process data across multiple contiguous .hdf5 files, first create a virtual v5_compatible.h5 file in the data directory containing the original
.hdf5 data files. This .v5 file allows the indexing of multiple .hdf5 files as a single dataset, enabling simpler and faster
data indexing and processing.

v5_compatible.h5 can be created by following these steps on a Treble, or a Linux machine with the correct dependencies:

1.  Run the `trebleconda` command in the terminal. This will set up the 'terra15py37' conda environment for data processing.
        - If the environment already exists, skip this step.
        - The list of installed environments can be checked by running `conda info --envs`
2.  Run: `conda activate terra15py37`
        - This will activate the environment.
3.  Run: `create_v5_compatible_dataset.py --directory <DIRECTORY> --data_product data_product`
        <DIRECTORY> should be replaced with the path to the directory containing data to be processed.
        This will create a file in <DIRECTORY> called `v5_compatible.h5`. Use the path to this file for this script.
"""

# INPUT PARAMETERS
#####################################################################################################################

# Define the path to a previously created .h5 virtual hdf5 file or single .hdf5 file.
hdf5_data_file = "../sample_data/v5_compatible.h5"
# OR
# hdf5_data_file = "../sample_data/example_triggered_shot.hdf5"

# Define trigger processing parameters
CROP_DURATION = 1 # (s)
# The trigger reference file must contain a column named "Time", containing a list of trigger time strings.
# Trigger time strings are assumed to be defined in UTC.
trigger_reference_file = "split_dataset_triggers.csv"
input_time_string_format = " %Y/%m/%d %H:%M:%S.%f"

######################################################################################################################

import h5py
import copy
import pytz
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convert_velocity_to_strainrate(data, gauge_length_m, dx):
    gauge_samples = int(round(gauge_length_m / dx))
    return (data[:, gauge_samples:] - data[:, :-gauge_samples]) / (gauge_samples * dx)


def load_strainrate_data(hdf_path, duration_seconds):
    with h5py.File(hdf_path, "r") as f:
        metadata = dict(f.attrs)
        t2 = int(duration_seconds // metadata["dt_computer"]) + 1
        cropped_data = f["data_product"]["data"][:t2]

        if metadata["data_product"] == "velocity":
            cropped_data = convert_velocity_to_strainrate(cropped_data, metadata["pulse_length"], metadata['dx'])

        return cropped_data, metadata


def isoformat_timestamp(timestamp: float) -> str:
    """Python isoformat doesn't include Zulu suffix"""
    return f"{datetime.utcfromtimestamp(timestamp).isoformat()}Z"


def resave_triggered_data(src_hdf, dest_hdf, duration_seconds, trigger_timestamp=None):
    with h5py.File(src_hdf, "r") as src_file:
        src_attrs = dict(src_file.attrs)

        if trigger_timestamp:
            t1 = int((trigger_timestamp - src_attrs['saving_start_computer_time']) // src_attrs["dt_computer"])
        elif "trigger_start_line" in list(src_file.attrs.keys()):
            t1 = src_attrs['trigger_start_line']
            # if t1 == -1: t1=0
        else:
            raise AttributeError(
                f"No trigger_start_line found in file attributes. Unable to resave {src_hdf}"
            )
        t2 = t1 + int(duration_seconds // src_attrs["dt_computer"]) + 1


        assert src_attrs["nt"] >= t2, (
            f"File is not long enough to crop {duration_seconds}s worth of lines, "
            f"time after trigger is: {(src_attrs['nt'] - src_attrs['trigger_start_line']) * src_attrs['dt_computer']:.2f}s"
        )

        with h5py.File(dest_hdf, "w") as dest_file:
            for src_group_name, src_group in src_file.items():
                dest_group = dest_file.create_group(src_group_name)
                for source_dset_name, source_dset in src_group.items():

                    # crops data to trigger
                    if source_dset_name in ["data", "gps_time", "posix_time"]:
                        data = source_dset[t1:t2]
                    else:
                        data = source_dset

                    # adds data and attributes to new group
                    dest_dataset = dest_group.create_dataset(source_dset_name, data=data, dtype=source_dset.dtype)
                    dest_dataset.attrs.update(source_dset.attrs)

            # rewrite top level attributes
            if "gps_time" in src_file["data_product"].keys():
                new_start_time_gps = src_file["data_product"]["gps_time"][t1]
            else:
                new_start_time_gps = 0

            new_start_time_computer = src_file["data_product"]["posix_time"][t1]

            src_updated = copy.deepcopy(src_attrs)
            src_updated.update(
                {
                    "trigger_start_line": 0,
                    "file_start_computer_time": new_start_time_computer,
                    "file_start_computer_time_string": isoformat_timestamp(new_start_time_computer),
                    "file_start_gps_time": new_start_time_gps,
                    "nframes_allocated": 0,
                    "nframes_occupied": 0,
                    "nt": t2 - t1,
                }
            )
            dest_file.attrs.update(src_updated)


def plot_data(
        data,
        metadata,
        title,
        image_path: str
):
    plt.figure()
    pos_end = metadata["sensing_range_end"]
    pos_start = metadata["sensing_range_start"]
    sample_rate = 1/metadata['dt_computer']

    plt.title(title, fontsize=20)
    plt.imshow(
        data,
        aspect="auto",
        cmap="gray",
        extent=(pos_start, pos_end, 1 / sample_rate * data.shape[0], 0),
        vmin=-3 * np.std(data),
        vmax=3 * np.std(data),
        interpolation="none",
    )
    plt.xlabel("Position (m)")
    plt.ylabel("Time from Trigger (s)")
    plt.savefig(image_path)
    plt.close()


# Get file time boundaries
with h5py.File(hdf5_data_file, "r") as f:
    t_min = f["data_product"]["posix_time"][0]
    t_max = f["data_product"]["posix_time"][-1]

print(f"Data Start Time (UTC): {datetime.utcfromtimestamp(t_min).strftime(input_time_string_format)}\n"
      f"Data Stop Time  (UTC): {datetime.utcfromtimestamp(t_max).strftime(input_time_string_format)}")

# get trigger timestamps
trigger_data = pd.read_csv(trigger_reference_file)
trigger_times = list(trigger_data["Time"])
trigger_datetimes = [datetime.strptime(t, input_time_string_format) for t in trigger_times]
trigger_datetimes = [t.replace(tzinfo=pytz.utc) for t in trigger_datetimes]

# crop and resave data from each trigger timestamp
for t in trigger_datetimes:
    timestring = t.strftime("YMD%Y%m%d-HMS%H%M%S.%f")

    if t_min < t.timestamp() < t_max:
        try:
            print(f"Processing trigger : {t.strftime(input_time_string_format)}")
            output_filename = timestring + "_cropped_data.hdf5"

            resave_triggered_data(hdf5_data_file, output_filename, CROP_DURATION, trigger_timestamp=t.timestamp())

            # Checks data validity by re-loading and plotting.
            data, md = load_strainrate_data(output_filename, CROP_DURATION)
            plot_data(data, md, title=timestring, image_path=timestring + "_cropped_data.png")
        except Exception as e:
            print(f"Could not process data for trigger: {t.strftime(input_time_string_format)}")
            print(e)


    else:
        print(f"Trigger time {t.strftime(input_time_string_format)} is outside time limits of file.")
