"""Demo script to load velocity data and spatially filter to 'local velocity'. Does not resave data."""

import h5py
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

### Set parameters ###############################################################
fname = r"../sample_data/example_triggered_shot.hdf5"
T_DURATION = 2 # (s)
FILTER_LENGTH = 50 # (m) Length of the local velocity filter. Wavelengths above <FILTER_LENGTH> will be filtered out.
##########################################################################################

def convert_velocity_to_strainrate(data, gauge_length_m, dx):
    gauge_samples = int(round(gauge_length_m / dx))
    return (data[:, gauge_samples:] - data[:, :-gauge_samples]) / (gauge_samples * dx)


def simple_load_data(hdf_path, duration_seconds):
    with h5py.File(hdf_path, "r") as f:
        metadata = dict(f.attrs)
        t2 = int(duration_seconds // metadata["dt_computer"]) + 1
        cropped_data = f["data_product"]["data"][:t2]

        return cropped_data, metadata


def plot_data(
        data,
        metadata,
        title,
        image_path: str = None,
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
    plt.ylabel("Time (s)")
    if image_path is not None:
        plt.savefig(image_path)


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


velocity, metadata = simple_load_data(fname, T_DURATION)
local_velocity = convert_velocity_to_local_velocity(velocity, dx=metadata['dx'], velocity_filter_length=FILTER_LENGTH)

plot_data(velocity, metadata, "Velocity Data", image_path="convert_to_local_velocity_1.png")
plot_data(local_velocity, metadata, "Local Velocity Data", image_path="convert_to_local_velocity_2.png")
plt.show()