"""
Example script to plot correctly weighted Power Spectral Density of Treble hdf5 data.
Uses 2 methods to calculate the PSD:
1) PSD of the mean signal
2) Mean of the PSDs.
"""


### Set parameters ###############################################################
hdf_file = "../sample_data/example_triggered_shot.hdf5"
gauge_length = 20  # (meters)
##########################################################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import h5py
from datetime import datetime


def load_hdf_slice(filepath, t_start=None, t_duration=None, x_start=None, x_stop=None, info=True):
    """
        Loads data and metadata from .hdf5 file. Optionally slices data in time and space.

        Args:
            file_path: hdf5 file path.
            t_start: Time from start of data to begin slice (s)
            t_duration: Length of desired data slice (s)
            x_start: Start fibre distance from front of Treble (m)
            x_stop: Stop fibre distance from front of Treble (m)

        Returns:
            data: Sliced data array.
            metadata: Attributes of hdf file.
            timestamps: Sliced timestamp vector.
            distances: Sliced distance vector.

        """
    print(f"Loading: {filepath}")

    with h5py.File(filepath, "r") as f:
        md = dict(f.attrs)
        dt = md["dt_computer"]
        dx = md["dx"]

        # Accesses full data but does not load into memory
        data = f["data_product"]["data"]
        if "gps_time" in f["data_product"].keys():
            t = f["data_product"]["gps_time"]
        else:
            print("Using Computer Time, either because no GPS data was recorded or this is a re-saved dataset.")
            t = f["data_product"]["posix_time"]
        x = np.arange(md["nx"]) * md["dx"] + md["sensing_range_start"]

        if info is True:
            # Prints dimensions of full data
            print("Full Dataset Properties:")
            print(f"    Data Shape:         {data.shape}")
            print(f"    t_end - t_start:    {t[-1]-t[0]:.8f} s")
            print(f"    nt * dt_computer:   {t.shape[0] * dt}")
            print(f"    Timestamps:         [{t[0]:.2f} : {t[-1]:.2f}] ({t[-1] - t[0]:.2f}s)")
            print(f"    Times (UTC):        [{datetime.utcfromtimestamp(t[0])} : {datetime.utcfromtimestamp(t[-1])}]")
            print(f"    Distance:           [{x[0]:.1f} : {x[-1]:.1f}] m")

        # Calculates slice boundaries in samples
        if t_start:
            t1 = int(t_start / dt)
        else:
            t1 = 0
        if t_duration:
            t2 = t1 + int(t_duration / dt)
            # Enforces limit on end of data.
            if t2 > len(t):
                t2 = len(t)
        else:
            t2 = None

        if x_start:
            x1 = int((x_start-md["sensing_range_start"])/dx)
        else:
            x1 = None
        if x_stop:
            x2 = int((x_stop-md["sensing_range_start"])/dx)
        else:
            x2 = None

        # Slices data in space and time.
        # Only the SLICE of data is loaded into memory
        data = data[t1:t2, x1:x2]
        t = t[t1:t2]
        x = x[x1:x2]

        if info is True:
            # Prints dimensions of sliced output data
            print("Loading data slice:")
            print(f"    Data Shape:         {data.shape}")
            print(f"    t_end - t_start:    {t[-1]-t[0]:.8f} s")
            print(f"    nt * dt_computer:   {t.shape[0] * dt}")
            print(f"    Timestamps:         [{t[0]:.2f} : {t[-1]:.2f}] ({t[-1] - t[0]:.2f}s)")
            print(f"    Times (UTC):        [{datetime.utcfromtimestamp(t[0])} : {datetime.utcfromtimestamp(t[-1])}]")
            print(f"    Distance:           [{x[0]:.1f} : {x[-1]:.1f}] m")
        return data, md, t, x

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


def plot_psd(psd, frequencies, title=None, label=None, ax=None):
    # plots on existing axis
    if ax:
        plt.sca(ax)
    else:
        # otherwise creates figure
        plt.figure(figsize=(10,6))

    plt.plot(frequencies, psd, linewidth=0.7, label=label)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("$(strain/s)^2$/Hz")
    plt.grid(visible=True, which="both")



# loads data
data, metadata, t, x = load_hdf_slice(
    hdf_file
)

# converts to strain rate if required
if metadata["data_product"] == "velocity":
    data = convert_velocity_to_strainrate(data, gauge_length, metadata["dx"])
    x = correct_gauge_length_offset(x, gauge_length)

# AVERAGE SIGNAL ACROSS ALL SPATIAL LOCATIONS, THEN CALCULATE PSD
mean_signal = np.mean(data, axis=1)
f, psd_of_mean_signal = welch(mean_signal, fs=1/metadata["dt_computer"], axis=0, nperseg=len(data), noverlap=0)
plot_psd(
    psd_of_mean_signal,
    f,
    label="AVERAGE SIGNAL IN SPACE, then CALCULATE PSD"
)

# CALCULATE PSD FOR EACH SPATIAL LOCATION, THEN AVERAGE PSD ACROSS ALL SPATIAL LOCATIONS
f, psd_2d = welch(data, fs=1/metadata["dt_computer"], axis=0, nperseg=len(data), noverlap=0)
mean_psd = np.mean(psd_2d, axis=1)
plot_psd(
    mean_psd,
    f,
    label="CALCULATE PSD AT ALL SPATIAL LOCATIONS, then AVERAGE PSDs",
    ax=plt.gca()
)
plt.suptitle(hdf_file)
plt.title("Power Spectral Density (PSD) of Strain Rate")
plt.legend()
plt.tight_layout()
plt.savefig("plot_hdf5_PSD.png")
plt.show()