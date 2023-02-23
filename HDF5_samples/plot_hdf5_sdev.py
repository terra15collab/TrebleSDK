"""Example for loading + plotting Terra15 Treble .hdf5 data."""

import h5py
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
            print(f"Full Dataset Properties:")
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
            print(f"Loading data slice:")
            print(f"    Data Shape:         {data.shape}")
            print(f"    t_end - t_start:    {t[-1]-t[0]:.8f} s")
            print(f"    nt * dt_computer:   {t.shape[0] * dt}")
            print(f"    Timestamps:         [{t[0]:.2f} : {t[-1]:.2f}] ({t[-1] - t[0]:.2f}s)")
            print(f"    Times (UTC):        [{datetime.utcfromtimestamp(t[0])} : {datetime.utcfromtimestamp(t[-1])}]")
            print(f"    Distance:           [{x[0]:.1f} : {x[-1]:.1f}] m")
        return data, md, t, x


def convert_velocity_to_strainrate(velocity, dx, gauge_length):
    """Convert velocity data to strainrate by performing gauge calculation."""
    gauge_samples = int(round( gauge_length / dx ))
    gauge_length  = gauge_samples * dx
    strain_rate = velocity[:, gauge_samples:] - velocity[:, :-gauge_samples]
    strain_rate = strain_rate / gauge_length
    return strain_rate


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


def calculate_sdev_records(data_array, t, t_sdev):
    """
        Calculates standard deviation over time sections of 2d array.
        Args:
            data_array: 2d data array
            t: Time array corresponding to input data
            t_sdev: Time per RMS section

        Returns:
            sdev_array: Sdev Data
            t_new: Subsampled time vector corresponding to sdev_array
        """
    # calculates new index
    dt = np.mean(np.diff(t))
    nt, nx = data_array.shape
    sdev_samples = int(np.ceil(t_sdev / dt))
    index = np.arange(0, nt, sdev_samples)

    # processes data in sections
    sdev_array = np.zeros((len(index), nx))
    for i, k in enumerate(index):
        data_section = data_array[k: k + sdev_samples]
        sdev_array[i] = np.std(data_section, axis=0)

    # compensates for time shift caused by sectioning data
    t_new = t[index] + t_sdev / 2
    return sdev_array, t_new


def plot_2d_heatmap(data_array, t_vector, x_vector, title=None, ax=None, **kwargs):
    """Plots heatmap of 2d data array."""
    try:
        # formats timestamps nicely as datetime objects
        t_vector = [datetime.utcfromtimestamp(t) for t in t_vector]

        # allows for custom color map
        if "cmap" in kwargs.keys():
            cmap = kwargs["cmap"]
        else:
            cmap="Greys"

        # parameters for color scaling
        data_mean = np.mean(data_array)
        data_sdev = np.std(data_array)
        c_max = data_mean + 4 * data_sdev
        c_min = data_mean - 4 * data_sdev

        # creates a figure if not given an existing axis
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.subplot(111)
        else:
            plt.sca(ax)

        # plots figure
        plt.pcolormesh(
            x_vector,
            t_vector,
            data_array,
            cmap=cmap,
            vmin=c_min,
            vmax=c_max,
            shading="auto")
        plt.colorbar()
        # flips time axis to read top->bottom
        ax.invert_yaxis()
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.title(title)
        plt.xlabel("Distance (m)")
        plt.ylabel("Time (UTC)")
    except Exception as e:
        print("Failed to plot data heatmap.")
        print(e)


# Define the location of the .hdf5 file
hdf_file = "../sample_data/example_triggered_shot.hdf5"
# Define the gauge length
gauge_length = 10  # (meters)
# Define time over which to calculate standard deviation summary data
t_sdev = 0.01 # (s)

# Read data, cropped from 0.5 seconds into the file until 2 seconds in
# and only between locations 1950 - 2050 meters.
velocity_data, metadata, t, x = load_hdf_slice(
    hdf_file,
    t_start=0.5,
    t_duration=2,
    x_start=0,
    x_stop=900
)

# convert velocity data to strainrate, using a custom gauge_length
dx = x[1] - x[0]
strainrate_data = convert_velocity_to_strainrate(velocity_data, dx, gauge_length)
x_strainrate = correct_gauge_length_offset(x, gauge_length)

# calculate std deviation of strainrate, in <t_sdev>-second-long sections
sdev_data, t_new = calculate_sdev_records(strainrate_data, t, t_sdev)

# create figure
plt.figure(figsize=(10,6))
plt.suptitle(hdf_file)

# plots both data types on the same plot
ax1 = plt.subplot(211)
plot_2d_heatmap(
    velocity_data, t, x,
    ax=ax1,
    title=f"Velocity Data",
    cmap="seismic"
)

ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
plot_2d_heatmap(
    sdev_data, t_new, x_strainrate,
    ax=ax2,
    title=f"Std. Dev of Strainrate. Gauge Length = {gauge_length}m",
    cmap="viridis"
)

ax1.invert_yaxis()
plt.tight_layout()
plt.savefig("plot_hdf5_sdev.png")
plt.show()