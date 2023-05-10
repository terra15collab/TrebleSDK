"""
Crops and resaves a triggered Terra15 .hdf5 data file.
"""
import copy
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


### Set parameters ###############################################################
original_filename = "../sample_data/example_triggered_shot.hdf5"
CROP_DURATION = 1 # (s)
PLOT_DURATION = 2 # (s)
##########################################################################################



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


def resave_triggered_data(src_hdf, dest_hdf, duration_seconds):
    with h5py.File(src_hdf, "r") as src_file:
        src_attributes = dict(src_file.attrs)

        # get indices for time crop
        if "trigger_start_line" in list(src_file.attrs.keys()):
            t1 = src_attributes['trigger_start_line']
            t2 = t1 + int(duration_seconds // src_attributes["dt_computer"]) + 1
        else:
            raise AttributeError(
                f"No trigger_start_line found in file attributes. Unable to resave {src_hdf}"
            )

        assert src_attributes["nt"] >= t2, (
            f"File is not long enough to crop {duration_seconds}s worth of lines, "
            f"time after trigger is: {(src_attributes['nt'] - src_attributes['trigger_start_line']) * src_attributes['dt_computer']:.2f}s"
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
            if src_file["data_product"]["gps_time"]:
                new_start_time_gps = src_file["data_product"]["gps_time"][t1]
            else:
                new_start_time_gps = 0

            new_start_time_computer = src_file["data_product"]["posix_time"][t1]

            src_updated = copy.deepcopy(src_attributes)
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


def plot_data(data, t, x, title=None, units=None, axis=None, cmap="gray"):
    t_start = datetime.utcfromtimestamp(t[0])
    t_rel = t - t[0]

    if axis is not None:
        plt.sca(axis)
    else:
        plt.figure(figsize=(8, 6))

    if title is not None:
        plt.suptitle(title, fontsize=12)

    plt.title(t_start, loc="left", fontsize=10)

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


if __name__ == "__main__":

    # resaves data cropped to trigger location
    new_filename = original_filename + "_cropped.hdf5"
    resave_triggered_data(original_filename, new_filename, CROP_DURATION)

    raw_data, md_raw, t_raw, x_raw = simple_load_data(original_filename, PLOT_DURATION)
    raw_data = convert_velocity_to_strainrate(raw_data, md_raw['pulse_length'], md_raw['dx'])
    x_raw = correct_gauge_length_offset(x_raw, md_raw['pulse_length'])

    cropped_data, md_crop, t_crop, x_crop = simple_load_data(new_filename, PLOT_DURATION)
    cropped_data = convert_velocity_to_strainrate(cropped_data, md_crop['pulse_length'], md_crop['dx'])
    x_crop = correct_gauge_length_offset(x_crop, md_crop['pulse_length'])

    plot_data(raw_data, t_raw, x_raw, "Uncropped Data", units="strainrate (strain/s)")

    trigger_offset = md_raw["trigger_start_line"] * md_raw["dt_computer"]
    # label trigger locations
    plt.axhline(0, color="k")
    plt.annotate("File Start", (x_raw[0], 0.075))
    plt.axhline(trigger_offset, color="r", linestyle=":")
    plt.annotate(f"Trigger Time = {trigger_offset*1e3:.0f}ms", (x_raw[0], trigger_offset))
    plt.axhline(trigger_offset+CROP_DURATION, color="r", linestyle=":")
    plt.annotate(f"Trigger Time + {CROP_DURATION}s", (x_raw[0], trigger_offset + CROP_DURATION))
    plt.savefig('crop_data_to_trigger_1.png')

    plot_data(cropped_data, t_crop, x_crop, f"Cropped to Trigger + {CROP_DURATION}s", units="strainrate (strain/s)")
    plt.savefig('crop_data_to_trigger_2.png')
    plt.show()