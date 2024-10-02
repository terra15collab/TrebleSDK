"""
Convert a terra15 HDF5 file containing velocity data to a strainrate dataset with a specified gauge length and high-pass filter.
Generates a new HDF5 file containing converted strainrate data.
"""

#####################################################################################################################
# File to convert. Must be a Terra15 Treble .hdf5 data file in units of velocity.
src_file = "../sample_data/example_triggered_shot.hdf5"
GAUGE_LENGTH = 5 # (m)
######################################################################################################################

import h5py
import numpy as np
import matplotlib.pyplot as plt


def convert_velocity_to_strainrate(data, gauge_length_m, dx):
    gauge_samples = int(round(gauge_length_m / dx))
    exact_gauge = gauge_samples * dx
    return (data[:, gauge_samples:] - data[:, :-gauge_samples]) / exact_gauge, exact_gauge


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


def convert_hdf5_to_strainrate(src_file, gauge_length_m):
    if src_file.endswith('.hdf5'):
        dst_file = src_file.replace(".hdf5", f'converted_strainrate_{gauge_length_m}m_gauge.hdf5')
    else:
        raise ValueError("Source file must be an HDF5 file")
    print(f"Converting {src_file} to strainrate with GAUGE {gauge_length_m} m")

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
                            velocity_data = source_dset[:]
                            strainrate_data, L_gauge = convert_velocity_to_strainrate(velocity_data, gauge_length_m, dx)

                            # Create the strainrate dataset
                            dest_dataset = dest_group.create_dataset(source_dset_name, data=strainrate_data)
                            # Copy the attributes from the velocity dataset
                            vel_attrs = dict(source_dset.attrs)
                            dest_dataset.attrs.update(vel_attrs)

                        else:
                            print(f"    Copying dataset: {source_dset_name}")
                            # Copies other datasets exactly
                            dest_dataset = dest_group.create_dataset(source_dset_name, data=source_dset, dtype=source_dset.dtype)
                            dest_dataset.attrs.update(source_dset.attrs)

                # Update the top-level attributes to strainrate
                dst_hdf.attrs['data_product'] = 'strainrate'
                dst_hdf.attrs['data_product_units'] = '1/s'
                dst_hdf.attrs['gauge_length'] = L_gauge

            print(f"Converted data saved in {dst_file}")

            return dst_file


# Test load the original data
data, md, t, x = simple_load_data(src_file)
plot_data(data, t, x,
          title=f"Velocity Data\n"
                f"Pulse Length = {md['pulse_length']:.2f}m",
          fname="convert_hdf5_velocity_to_strainrate_1.png")

# Perform the copy and transformation
dst_file = convert_hdf5_to_strainrate(src_file, GAUGE_LENGTH)

# Test load the converted data
data, md, t, x = simple_load_data(dst_file)
plot_data(data, t, x,
          title=f"File converted to strainrate\n"
                f"Pulse Length = {md['pulse_length']:.2f}m, Gauge Length={md['gauge_length']:.2f}m",
          fname="convert_hdf5_velocity_to_strainrate_2.png")

plt.show()