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


def plot_2d(data, hor, vert, axis=None, cmap=None, **kwargs):

    extent = min(hor), max(hor), max(vert), min(vert)

    # parameters for color scaling
    data_mean = np.mean(data)
    data_sdev = np.std(data)

    # specifies colour parameters:
    if "vmax" in kwargs.keys():
        vmax = kwargs["vmax"]
    else:
        vmax = data_mean + 4 * data_sdev

    if "vmin" in kwargs.keys():
        vmin = kwargs["vmin"]
    else:
        vmin = data_mean - 4 * data_sdev

    if cmap is None:
        cmap = plt.cm.seismic.copy()

    # creates a figure if not given an existing axis
    if axis:
        plt.sca(axis)
    else:
        plt.figure(figsize=(10, 6))
        axis = plt.subplot(111)

    plt.imshow(
        data,
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        extent=extent,
        vmin=vmin, vmax=vmax,
        aspect="auto"
    )

    plt.colorbar()
    axis.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xlabel("Distance (m)")
    plt.ylabel("Time (UTC)")

# Define the location of the .hdf5 file
hdf_file = "../sample_data/example_triggered_shot.hdf5"
# Define the gauge length
gauge_length = 10  # (meters)

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
dx = metadata["dx"]
strainrate_data = convert_velocity_to_strainrate(velocity_data, dx, gauge_length)
x_strainrate = correct_gauge_length_offset(x, gauge_length)

# converts timestamps to plot-compatible format.
t_plot = np.array(t, dtype="datetime64[s]")
t_plot = mdates.date2num(t_plot)

# create figure
plt.figure(figsize=(10,6))
plt.suptitle(hdf_file)

# plots both data types on the same plot
ax1 = plt.subplot(211)
plt.title("Velocity Data")
plot_2d(
    velocity_data, x, t_plot,
    axis=ax1,
    cmap="seismic"
)

ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
plt.title(f"Strainrate Data. Gauge Length = {gauge_length}m")
plot_2d(
    strainrate_data, x_strainrate, t_plot,
    axis=ax2,
    cmap="viridis"
)
plt.xlim(x[0],x[-1])

plt.tight_layout()
plt.savefig("plot_hdf5_simple.png")
plt.show()