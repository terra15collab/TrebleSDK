"""Demo script to load strainrate data and convert it to strain units."""

import h5py
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter, sosfiltfilt
import numpy as np
from datetime import datetime

### Set parameters ###############################################################
fname = r"../sample_data/example_triggered_shot.hdf5"
T_DURATION = 2 # (s)
# High-pass filter cutoff frequency (Hz) applied to the strainrate data before integration to strain.
F_HP = 4 # (Hz)
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

def convert_strainrate_to_strain(strainrate, dt, f_hp=0.1):
    # High-pass filter data at cutoff frequency f_hp
    strainrate = highpass_filter(strainrate, f_hp, dt)

    # Integrate filtered strainrate to strain
    omega_0 = 2*np.pi*f_hp
    b = np.array([dt, 0], dtype=strainrate.dtype)
    a = np.array([1 + omega_0*dt,-1], dtype=strainrate.dtype)
    strain = lfilter(b, a, strainrate, axis=0)
    return strain


def highpass_filter(data, f_hp, dt):
    fs = 1/dt
    coeff = butter(1, f_hp, btype="hp", fs=fs, output="sos")
    data = sosfiltfilt(coeff, data, axis=0)
    return data.astype(data.dtype)

velocity, metadata, t, x = simple_load_data(fname, T_DURATION)

strainrate = convert_velocity_to_strainrate(velocity, metadata['pulse_length'], metadata['dx'])
x = correct_gauge_length_offset(x, metadata['pulse_length'])
strain = convert_strainrate_to_strain(strainrate, metadata['dt_computer'], F_HP)

plot_data(strainrate, t, x, "Strainrate Data", units="strainrate (strain/s)")
plt.savefig("convert_to_strain_1.png")
plot_data(strain, t, x, "Strain Data", units="strain")
plt.savefig("convert_to_strain_2.png")
plt.show()