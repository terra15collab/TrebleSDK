% By default, Terra15 HDF5 files are written in the HDF5 1.10 version
% format. Matlab is still only compatible with HDF5 1.8 version files. To
% convert 1.10 files into 1.8 files, please run the "h5format_convert" tool
% on the HDF5 files first, so they can be read by Matlab. The
% "h5format_convert" tool is part of the HDF5 tools package that can be
% downloaded here: https://confluence.hdfgroup.org/display/support/HDF5+1.10.7#files

% Define the location of the HDF5 file:
hdf5_filepath = "~/Documents/samplehdf/v5_compatible.h5"

% Define the gauge_length:
gauge_length = 50; % [meters]

% Read the velocity data, from 2 seconds into the file until 4 seconds in, and only between locations 2100 and 2500m:
[velocity_data, metadata, t, x] = load_hdf5_slice(hdf5_filepath, 2, 4, 2100, 2500);

% Calculate strain_rate, with a gaugelength as set by gauge_length
strainrate_data = convert_velocity_to_strainrate(velocity_data, metadata, gauge_length);
% Calculate the distance vector for strain_rates:
x_strainrate = correct_gauge_length_offset(x, gauge_length)


% Plot Velocity
ax1 = subplot(2,1,1)
v = pcolor(x, t, velocity_data')
set(v, 'EdgeColor', 'none')
colormap('gray')
axis ij


% Plot Strainrates:
ax2 = subplot(2,1,2)
s = pcolor(x_strainrate, t, strainrate_data')
set(s, 'EdgeColor', 'none')
colormap('gray')
axis ij
linkaxes([ax1,ax2], 'x')


function result = getAttributeValue(attrs, name)
    % convenience function for readingvalues from attributes in an HDF5
    % file
    idx = find(strcmp({attrs.Name}, name))
    result = attrs(idx).Value
end
function timestamps = get_timestamps_from_hdf5(file_path)
    % Returns timestamp vector from .hdf5 data file.
    nt = double(h5readatt(file_path,'/', "nt"));  % counts needs to be a "double", not an int.
    timestamps = h5read(file_path, '/data_product/posix_time', 1, nt);  % reading only the first nt values, incase there's zero-padding
    timestamps=datetime(timestamps, 'ConvertFrom','posixtime');   % converting from posix time to matlab timestamps
    
    time_span = timestamps(end) - timestamps(1);
    fprintf('Total time: %8.2f seconds \n', seconds(time_span))
    fprintf('Start time: %s\n', datestr(timestamps(1),'HH:MM:SS.FFF'))
    fprintf('Stop time: %s\n', datestr(timestamps(end), 'HH:MM:SS.FFF'))
end
function distances = get_distances_from_hdf5(file_path)
    % Returns distance vector from .hdf5 data file.
    nx = double(h5readatt(file_path,'/', "nx"));  % counts needs to be a "double", not an int.
    dx = double(h5readatt(file_path,'/', "dx"));  % counts needs to be a "double", not an int.
    sensing_range_start = double(h5readatt(file_path,'/', "sensing_range_start")); 
    
    distances = linspace(0, nx-1, nx) * dx;
    distances = distances + sensing_range_start;
    
    distance_span = distances(end) - distances(1);
    
    fprintf('Total distance: %8.2f m \n', distance_span)
    fprintf('Start distance: %8.2f m \n', distances(1))
    fprintf('Stop distance: %8.2f m \n', distances(end))
end
function [data_array, metadata, timestamps, distances] = load_hdf5_slice(file_path, t_start, t_stop, x_start, x_stop)
%     Loads data and metadata from .hdf5 file. Optionally slices data in time and space.
%     Args:
%         file_path: hdf5 file path.
%         t_start: Time start relative to start of data (s)
%         t_stop: Time stop relative to start of data (s)
%         x_start: Start distance from front of Treble (m)
%         x_stop: Stop distance from front of Treble (m)
%     Returns:
%         data_array: Sliced section of dataset.
%         metadata: Attributes of save file.
%         timestamps: Sliced timestamp vector.
%         distances: Sliced distance vector.

    fprintf("Loading hdf5: %s \n", file_path)
    timestamps = get_timestamps_from_hdf5(file_path);
    distances = get_distances_from_hdf5(file_path);
    
    % extract metadata parameters
    dt = double(h5readatt(file_path,'/', "dt_computer"));  
    dx = double(h5readatt(file_path,'/', "dx"));  
    nt = double(h5readatt(file_path,'/', "nt"));  
    nx = double(h5readatt(file_path,'/', "nx"));  
    metadata=h5info(file_path, '/').Attributes;
    
    % defines allowed time/distance ranges
    t_relative = timestamps - timestamps(1);
    t_min = t_relative(1);
    t_max = t_relative(end);
    x_min = distances(1);
    x_max = distances(end);
    
    t1 = double(round( t_start / dt ))  +1;
    t2 = double(round( t_stop / dt ))   +1;
    x1 = double(round( (x_start - x_min) / dx )) +1;
    x2 = double(round( (x_stop - x_min) / dx ))  +1;
    
    data_array = h5read(file_path, '/data_product/data', [x1, t1], [x2-x1, t2-t1]);
    timestamps = timestamps(t1:t2-1);
    distances = distances(x1:x2-1);
    
    %confirming dimensions of output arrays:
    t_start = timestamps(1);
    t_stop = timestamps(end);
    x_start = distances(1);
    x_stop = distances(end);

    fprintf("Loaded data slice: \n")
    fprintf("    UTC Times: [%9.3f: %9.3f] (%.2f)\n", posixtime(t_start), posixtime(t_stop), seconds(t_stop-t_start))
    fprintf("    UTC Times: [%s: %s] (%.2fs)\n", t_start, t_stop, seconds(t_stop-t_start))
    fprintf("    UTC Times: [%s : %s]\n", datestr(t_start, 'eYYYY-mm-DD HH:MM:SS.FFF'), datestr(t_stop, 'YYYY-mm-DD HH:MM:SS.FFF'))
    fprintf("    Distance:    [%.1f : %.1f] m \n", x_start, x_stop)
    
end
function strain_rate = convert_velocity_to_strainrate(data_array, metadata, gauge_length)
%    Calculates strain_rate from velocity data, with a given gauge_length 
    dx = getAttributeValue(metadata, 'dx')
    gauge_samples = int64(round( gauge_length / dx ));
    converted_data = data_array(gauge_samples+1:end, 1:end) - data_array(1:end-gauge_samples, 1:end);
    strain_rate = converted_data / gauge_length;
end
function x_correct = correct_gauge_length_offset(x_vector, gauge_length)
    % Compensate for distance shift of data caused by gauge_length calculation.
    % crops end of x_vector by gauge length
    dx = x_vector(2) - x_vector(1);
    gauge_samples = int64(round(gauge_length / dx));
    x_correct = x_vector(1:end-gauge_samples);

    % compensates for GL/2 signal offset
    x_correct = x_correct + gauge_length / 2;
end

