"""Example script to extract and save spatially-averaged audio data from one or more Treble .hdf5 recordings."""

### Set parameters ###############################################################
data_directory = "../sample_data/"
t_start = 0.1 # (s)
t_duration = 1 # (s)
x_start = 0 # (m)
x_stop = 700 # (m)
##########################################################################################

import os
import numpy as np
import h5py
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import wavfile



def get_files(directory: str, filetype: str):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(filetype):
                file_list.append(os.path.join(root, file))
    file_list.sort()
    return file_list


def convert_velocity_to_strainrate(data, gauge_length_m, dx):
    gauge_samples = int(round(gauge_length_m / dx))
    return (data[:, gauge_samples:] - data[:, :-gauge_samples]) / (gauge_samples * dx)


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


hdf_files = get_files(data_directory, ".hdf5")

for filepath in tqdm(hdf_files):
    filename = os.path.split(filepath)[-1]
    data, metadata, t, x = load_hdf_slice(
        filepath,
        t_start=t_start,
        t_duration=t_duration,
        x_start=x_start,
        x_stop=x_stop,
        info=True
    )
    if metadata['data_product'] in ["velocity", "velocity_filtered"]:
        data = data - np.mean(data, axis=0)
        data = convert_velocity_to_strainrate(data, metadata['pulse_length'], metadata['dx'])

    mean_strainrate = np.mean(data, axis=1)
    audio_trace = mean_strainrate/np.abs(np.max(mean_strainrate))

    audio_filename = f"{os.path.splitext(filename)[0]}_audio"
    wavfile.write(
        audio_filename + ".wav",
        rate=int(1/metadata['dt_computer']),
        data=audio_trace
    )

    plt.figure(figsize=(8, 4))
    plt.title(f"Audio Data: {filename}")
    plt.plot(t-t[0], audio_trace)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(audio_filename+ ".png")
    plt.show()
