"""
Convert a terra15 HDF5 file containing velocity data to a strain dataset with a specified gauge length and high-pass filter.
Generates a new HDF5 file containing converted strain data.
"""

#####################################################################################################################
# File to convert. Must be a Terra15 Treble .hdf5 data file in units of velocity.
src_file = "../sample_data/example_triggered_shot.hdf5"
GAUGE_LENGTH = 5 # (m)
# High-pass filter cutoff frequency (Hz) applied to the strainrate data before integration to strain.
HIGH_PASS_FREQ = 4 # (Hz)
######################################################################################################################


import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def convert_velocity_to_strain(velocity, dx, dt, gauge_length_m, f_hp=0.1):
    # First convert velocity to strainrate
    strainrate, L_gauge = convert_velocity_to_strainrate(velocity, gauge_length_m, dx)

    # Returns strain and exact gauge length
    return convert_strainrate_to_strain(strainrate, dt, f_hp), L_gauge


def convert_velocity_to_strainrate(velocity, gauge_length_m, dx):
    n_gauge = int(round(gauge_length_m / dx))
    L_gauge = n_gauge * dx
    strainrate = (velocity[:, n_gauge:] - velocity[:, :-n_gauge]) / L_gauge
    # Returns strainrate and exact gauge length
    return strainrate, L_gauge


def convert_strainrate_to_strain(strainrate, dt, f_hp=0.1):
    # High-pass filter data at cutoff frequency f_hp
    strainrate = highpass_filter(strainrate, f_hp, dt)

    # Integrate filtered strainrate to strain
    omega_0 = 2*np.pi*f_hp
    b = np.array([dt, 0], dtype=strainrate.dtype)
    a = np.array([1 + omega_0*dt,-1], dtype=strainrate.dtype)
    strain = signal.lfilter(b, a, strainrate, axis=0)
    return strain


def highpass_filter(data, f_hp, dt):
    fs = 1/dt
    coeff = signal.butter(1, f_hp, btype="hp", fs=fs, output="sos")
    data = signal.sosfiltfilt(coeff, data, axis=0)
    return data.astype(data.dtype)


def plot_data(data, t, x, title, fname=None):
    plt.figure(figsize=(8, 6))

    t_rel = t - t[0]

    clim = np.max(np.abs(data))/10

    plt.title(title, loc="center", fontsize=12)

    plt.imshow(
        data,
        aspect="auto",
        cmap="gray",
        extent=(x[0], x[-1], t_rel[-1], t_rel[0]),
        vmin=-clim,
        vmax=clim,
        interpolation="none"
    )

    cbar = plt.colorbar()

    plt.xlabel("Fibre Distance (m)")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=300)


def simple_load_data(hdf_path):
    with h5py.File(hdf_path, "r") as f:
        metadata = dict(f.attrs)
        data = f["data_product/data"][:]
        t = f["data_product/gps_time"][:]
        x = metadata['sensing_range_start'] + np.arange(0, metadata['nx']) * metadata["dx"]
        return data, metadata, t, x


def convert_hdf5_to_strain(src_file, gauge_length_m, f_hp=0.1):
    if src_file.endswith('.hdf5'):
        dst_file = src_file.replace(".hdf5", f'converted_strainrate_{gauge_length_m}m_gauge.hdf5')
    else:
        raise ValueError("Source file must be an HDF5 file")
    print(f"Converting {src_file} to strain with GAUGE {gauge_length_m} m, HP filter {f_hp} Hz")

    # Check source file for velocity data
    with h5py.File(src_file, 'r') as src_hdf:
        if src_hdf.attrs['data_product'] != 'velocity':
            print(f"Source file {src_file} does not contain velocity data")
            return None

        else:
            # Recreate source file at destination location
            with h5py.File(dst_file, "w") as dst_hdf:
                # Copy top-level attributes
                dst_hdf.attrs.update(src_hdf.attrs)

                # Copy groups and datasets except the data_product dataset
                for src_group_name, src_group in src_hdf.items():
                    print(f"Copying group: {src_group_name}")
                    dest_group = dst_hdf.create_group(src_group_name)
                    for source_dset_name, source_dset in src_group.items():
                        if source_dset_name == "data":
                            # Copy and convert velocity dataset
                            print(f"    Copying and converting dataset: {source_dset_name}")
                            dx = dst_hdf.attrs['dx']
                            dt = dst_hdf.attrs['dt_computer']
                            velocity_data = source_dset[:]
                            strain_data, L_gauge = convert_velocity_to_strain(velocity_data, dx, dt, gauge_length_m, f_hp)

                            # Create the strain dataset
                            dest_dataset = dest_group.create_dataset(source_dset_name, data=strain_data, dtype=source_dset.dtype)
                            # Copy the attributes from the velocity dataset
                            vel_attrs = dict(source_dset.attrs)
                            dest_dataset.attrs.update(vel_attrs)

                        else:
                            print(f"    Copying dataset: {source_dset_name}")
                            # Copies other datasets exactly
                            dest_dataset = dest_group.create_dataset(source_dset_name, data=source_dset,
                                                                     dtype=source_dset.dtype)
                            dest_dataset.attrs.update(source_dset.attrs)

                # Update the top-level attributes to strain
                dst_hdf.attrs['data_product'] = 'strain'
                dst_hdf.attrs['data_product_units'] = 'm/m'
                dst_hdf.attrs['gauge_length'] = L_gauge

        print(f"Converted data saved in {dst_file}")

        return dst_file



# Test load the original data
data, md, t, x = simple_load_data(src_file)
plot_data(data, t, x,
          title=f"Velocity Data\n"
                f"Pulse Length = {md['pulse_length']:.2f}m",
          fname="convert_hdf5_velocity_to_strain_1.png")

print(f"Original data shape: {data.shape}")

# Perform the copy and transformation
dst_file = convert_hdf5_to_strain(src_file, GAUGE_LENGTH, HIGH_PASS_FREQ)

# Test load the converted data
data, md, t, x = simple_load_data(dst_file)
plot_data(data, t, x,
          title=f"File converted to strain.\n"
                f"Pulse Length = {md['pulse_length']:.2f}m, Gauge Length={md['gauge_length']:.2f}m, HP Filter={HIGH_PASS_FREQ}Hz",
          fname="convert_hdf5_velocity_to_strain_2.png")
print(f"Converted data shape: {data.shape}")

plt.show()