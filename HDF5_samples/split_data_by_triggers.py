"""
Script to crop out multiple sections of data from a continuous Treble dataset based on a list of ASCII time values.
Times are defined as ASCII strings in a .csv file.
Data output includes PRE_TRIGGER_DURATION seconds of data before the trigger time.

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

# Define the path to a .h5 virtual hdf5 file or .hdf5 file.
hdf5_data_file = "../sample_data/example_triggered_shot.hdf5"

# Define trigger processing parameters
POST_TRIGGER_DURATION = 1 # (s)
PRE_TRIGGER_DURATION = 0.1 # (s)
# The trigger reference file must contain a column named "Time", containing a list of trigger time strings.
# Trigger time strings are assumed to be defined in UTC.
trigger_reference_file = "split_data_triggers.csv"
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


def correct_gauge_length_offset(x_vector, gauge_length):
    """Compensate for distance shift of data caused by gauge_length calculation."""
    # crops end of x_vector by gauge length
    dx = x_vector[1] - x_vector[0]
    gauge_samples = int(round(gauge_length / dx))
    gauge_length = gauge_samples * dx
    x_correct = x_vector[:-gauge_samples]

    # compensates for GL/2 signal offset
    x_correct = x_correct + gauge_length / 2
    return x_correct


def simple_load_data(hdf_path, duration_seconds):
    with h5py.File(hdf_path, "r") as f:
        metadata = dict(f.attrs)
        t2 = int(duration_seconds // metadata["dt_computer"]) + 1
        cropped_data = f["data_product/data"][:t2]
        t = f["data_product/gps_time"][:t2]
        x = metadata['sensing_range_start'] + np.arange(0, metadata['nx']) * metadata["dx"]

        return cropped_data, metadata, t, x


def isoformat_timestamp(timestamp: float) -> str:
    """Python isoformat doesn't include Zulu suffix"""
    return f"{datetime.utcfromtimestamp(timestamp).isoformat()}Z"


def resave_custom_triggers(src_hdf, dest_hdf, trigger_timestamp, post_trigger_seconds, pre_trigger_seconds=0.):
    crop_duration = post_trigger_seconds + pre_trigger_seconds
    t_start = trigger_timestamp - pre_trigger_seconds

    with h5py.File(src_hdf, "r") as src_file:
        src_attrs = dict(src_file.attrs)

        # Converts timestamps to line numbers to crop
        if t_start < src_file["data_product/gps_time"][0]:
            raise ValueError(f"Trigger time {isoformat_timestamp(t_start)} is before start of file.")

        t1 = np.where(src_file["data_product/gps_time"][:] > t_start)[0][0]
        t2 = t1 + int(crop_duration // src_attrs["dt_computer"]) + 1

        # Determines location of trigger line within file to update attributes
        i_trigger = np.where(src_file["data_product/gps_time"][:] > trigger_timestamp)[0][0]
        trigger_start_line = i_trigger - t1

        assert src_attrs["nt"] >= t2, (
            f"File is not long enough to crop {crop_duration}s worth of lines, "
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
                    "trigger_start_line": trigger_start_line,
                    "trigger_start_time": trigger_timestamp,
                    "file_start_computer_time": new_start_time_computer,
                    "file_start_computer_time_string": isoformat_timestamp(new_start_time_computer),
                    "file_start_gps_time": new_start_time_gps,
                    "nframes_allocated": 0,
                    "nframes_occupied": 0,
                    "nt": t2 - t1,
                }
            )
            dest_file.attrs.update(src_updated)


def plot_data(data, t, x, title=None, units=None, axis=None, cmap="gray"):
    t_start = datetime.utcfromtimestamp(t[0])
    t_rel = t - t[0]

    if axis is not None:
        plt.sca(axis)
    else:
        plt.figure(figsize=(8, 6))

    if title is not None:
        plt.suptitle(title, fontsize=12)

    plt.title(f"UTC {t_start}", loc="left", fontsize=10)

    plt.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        extent=(x[0], x[-1], t_rel[-1], t_rel[0]),
        vmin=-4 * np.std(data),
        vmax=4 * np.std(data),
        interpolation="none"
    )

    cbar = plt.colorbar()
    if units is not None:
        cbar.set_label(units)

    plt.xlabel("Fibre Distance (m)")
    plt.ylabel("Time (s)")
    plt.tight_layout()


# Get file time boundaries
print(f"Checking time boundaries: {hdf5_data_file}")
with h5py.File(hdf5_data_file, "r") as f:
    t_min = f["data_product"]["gps_time"][0]
    t_max = f["data_product"]["gps_time"][-1]

print(f"Data Start Time (UTC): {datetime.utcfromtimestamp(t_min).strftime(input_time_string_format)}\n"
      f"Data Stop Time  (UTC): {datetime.utcfromtimestamp(t_max).strftime(input_time_string_format)}")

# get trigger timestamps
trigger_data = pd.read_csv(trigger_reference_file)

# crop and resave data from each trigger timestamp
for i, row in trigger_data.iterrows():
    trig_name = row["Name"]
    trig = datetime.strptime(row["Time"], input_time_string_format)
    trig = trig.replace(tzinfo=pytz.utc)

    if t_min < trig.timestamp() < t_max:
        try:
            print(f"Processing custom trigger : {trig.strftime(input_time_string_format)}")
            timestring = trig.strftime("UTC-YMD%Y%m%d-HMS%H%M%S.%f")

            output_filename = timestring + "_trigger_data.hdf5"

            resave_custom_triggers(
                hdf5_data_file,
                output_filename,
                trig.timestamp(),
                POST_TRIGGER_DURATION,
                pre_trigger_seconds=PRE_TRIGGER_DURATION
            )

            # Checks data validity by re-loading and plotting.
            data, md, t, x = simple_load_data(output_filename, POST_TRIGGER_DURATION + PRE_TRIGGER_DURATION)
            data = convert_velocity_to_strainrate(data, md['pulse_length'], md['dx'])
            x = correct_gauge_length_offset(x, md["pulse_length"])

            plot_data(data, t, x, title=output_filename, units="strainrate (strain/s)")
            plt.savefig(timestring + "_trigger_data.png")

            print(f"Successfully re-saved data for trigger: {trig.strftime(input_time_string_format)}")

        except Exception as e:
            print(f"Could not process data for trigger: {trig.strftime(input_time_string_format)}")
            print(e)


    else:
        print(f"Trigger time {trig.strftime(input_time_string_format)} is outside time limits of the file.")
