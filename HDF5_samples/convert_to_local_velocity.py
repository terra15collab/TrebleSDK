"""Demo script to load velocity data and spatially filter to 'local velocity'. Does not resave data."""

import h5py
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from datetime import datetime

### Set parameters ###############################################################
fname = r"../sample_data/example_triggered_shot.hdf5"
T_DURATION = 2 # (s)
FILTER_LENGTH = 50 # (m) Length of the local velocity filter. Wavelengths above <FILTER_LENGTH> will be filtered out.
##########################################################################################

def simple_load_data(hdf_path, duration_seconds):
    with h5py.File(hdf_path, "r") as f:
        metadata = dict(f.attrs)
        t2 = int(duration_seconds // metadata["dt_computer"]) + 1
        cropped_data = f["data_product/data"][:t2]
        t = f["data_product/gps_time"][:t2]
        x = metadata['sensing_range_start'] + np.arange(0, metadata['nx']) * metadata["dx"]

        return cropped_data, metadata, t, x


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


def convert_velocity_to_local_velocity(
    data: np.ndarray, dx: float, velocity_filter_length: float
):
    # Create a high-pass filter of the suitable length
    b = np.array([1.0, -1.0])
    k0 = 2 * np.pi / velocity_filter_length
    a = np.array([1.0 + k0 * dx / 2, -1.0 + k0 * dx / 2])
    # Apply in chunks to avoid memory issues
    chunk_size = 10000
    converted_data = data.copy()
    for i in range(0, data.shape[0], chunk_size):
        converted_data[i : i + chunk_size, :] = signal.filtfilt(
            b, a, data[i : i + chunk_size, :], axis=1
        )

    return converted_data


velocity, metadata, t, x = simple_load_data(fname, T_DURATION)
local_velocity = convert_velocity_to_local_velocity(velocity, dx=metadata['dx'], velocity_filter_length=FILTER_LENGTH)

plot_data(velocity, t, x, "Velocity Data", units='velocity (m/s)')
plt.savefig("convert_to_local_velocity_1.png")
plot_data(local_velocity, t, x, "Local Velocity Data", units='velocity (m/s)')
plt.savefig("convert_to_local_velocity_2.png")
plt.show()