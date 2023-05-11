"""Demo script to load strainrate data and convert it to strain units."""

import h5py
import matplotlib.pyplot as plt
from scipy.signal import lfilter, lfilter_zi
import numpy as np
from datetime import datetime

### Set parameters ###############################################################
fname = r"../sample_data/example_triggered_shot.hdf5"
T_DURATION = 2 # (s)
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


velocity, metadata, t, x = simple_load_data(fname, T_DURATION)

strainrate = convert_velocity_to_strainrate(velocity, metadata['pulse_length'], metadata['dx'])
x = correct_gauge_length_offset(x, metadata['pulse_length'])
strain = convert_strainrate_to_strain(strainrate, metadata['dt_computer'])

plot_data(strainrate, t, x, "Strainrate Data", units="strainrate (strain/s)")
plt.savefig("convert_to_strain_1.png")
plot_data(strain, t, x, "Strain Data", units="strain")
plt.savefig("convert_to_strain_2.png")
plt.show()