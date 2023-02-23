"""
Example script using PyQTGRAPH and Terra15 Treble client functions to plot the realtime waveform between two spatial locations.
"""

### Pyside6, Pyside2, or PyQt5 must be installed for correct display.
### Pyside6 is preferred.

### The "treble" API package must be installed to run the Treble API.
### Download links can be provided by Terra15.

### SETUP PARAMETERS ################################################################################
treble_ip = "10.0.0.225"
server_port = "48000"
n_frames = 10
update_rate = 0.16  # (s)
# Crops data between fibre index
x_start = 400
x_stop = 500
# Custom gauge length to convert velocity to strainrate
gauge_length = 5 # (m)
########################################################################################################

import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
from treble import acq_client

# setup Treble connection, initial acquisition to get metadata
client = acq_client.acq_Client()
client.connect_to_server(f"tcp://{treble_ip}:{server_port}")
_, md = client.fetch_data_product([-1],timeout=20000)

# Calculates x axis to display correct region.
x = np.arange(md['nx']) * md["dx"] + md["sensing_range"][0]

# Create plot widget
app = pg.mkQApp("Treble Time Series Example")
win = pg.GraphicsLayoutWidget(show=True, title=f"Streaming Treble Data: {treble_ip}")
plot1 = win.addPlot(title=f"Mean Strain Rate {x[x_start]:.0f}m->{x[x_stop]:.0f}m")
plot1.setLabel("bottom", text="time", units="samples")
plot1.setLabel("left", text=f"strain_rate")
curve = plot1.plot(pen='y')

# function to repeatedly acquire data and update plot
inc = 0
def update_plot():
    global inc, md
    data, md = client.fetch_data_product(
        list(range(-n_frames + 1, 1, 1)),
        timeout=20000,
        with_client_fn='convert_deformation_do_range_mean',
        client_fn_args={
            "start_index2": x_start,
            "end_index2": x_stop,
            "gauge_length": gauge_length,
            "dx_dec": md["dx"]
        }
    )
    data = data.reshape((data.shape[0] * data.shape[1], -1)).squeeze()

    curve.setData(data)

    ## Stop auto-scaling after the first data set is plotted
    if inc == 0:
        plot1.enableAutoRange('xy', False)
    inc += 1

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
# Increment timer every 160ms
# This is the smallest Treble frame size - no point trying to grab Treble data more frequently than this.
timer.start(update_rate * 1000)

if __name__ == '__main__':
     pg.exec()
