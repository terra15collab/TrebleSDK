"""Demo script to load strainrate data and convert it to strain units."""

import h5py
import matplotlib.pyplot as plt
from scipy.signal import lfilter, lfilter_zi
import numpy as np

### Set parameters ###############################################################
fname = r"../sample_data/example_triggered_shot.hdf5"
T_DURATION = 2 # (s)
##########################################################################################

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


def convert_strainrate_to_strain(data, dT):
    strain_scalar = dT
    leaky_characteristic_period = 0.4

    # Construct leaky integration filter to convert strain to strain rate
    b = 0.5 * strain_scalar * np.array([dT, dT])
    w0 = 2 * np.pi / leaky_characteristic_period
    a = np.array([w0 * dT / 2 + 1, w0 * dT / 2 - 1])
    zi=lfilter_zi(b, a)*data[0]

    filtered_data, z_out = lfilter(b, a, data, axis=0, zi=zi[:,None].T)

    return filtered_data


strainrate, metadata = load_strainrate_data(fname, T_DURATION)

strain = convert_strainrate_to_strain(strainrate, metadata['dt_computer'])

plot_data(strainrate, metadata, "Strainrate Data", image_path="convert_to_strain_1.png")
plot_data(strain, metadata, "Strain Data", image_path="convert_to_strain_2.png")
plt.show()