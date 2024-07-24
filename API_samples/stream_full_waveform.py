"""
Example script using PYQTGRAPH and Client Functions to plot full raw waveform Terra15 Treble data.
"""

### Pyside6, Pyside2, or PyQt5 must be installed for correct display.
### Pyside6 is preferred.

### The "treble" API package must be installed to run the Treble API.
### Download links are located in the README.md file.

### SETUP PARAMETERS ################################################################################
treble_ip = "localhost"
server_port = "48000"
n_frames = 1
update_rate = 0.16  # (s)
# Crops data between fibre index
x_start = 10
x_stop = 100
########################################################################################################

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from treble import acq_client

# setup Treble connection
client = acq_client.acq_Client()
client.connect_to_server(f"tcp://{treble_ip}:{server_port}")
_, md = client.fetch_data_product([-1],timeout=20000)

# Create window
app = pg.mkQApp("Plotting Example")
win = pg.GraphicsLayoutWidget(title=f"Streaming Treble Data: {treble_ip}")
win.setWindowTitle(f"Streaming Treble Data: {treble_ip}")

# Create date axis to correctly plot timestamps
date_axis = pg.DateAxisItem(orientation="left", utcOffset=0)
# Create image item to update inside plot
i1 = pg.ImageItem()

# Add a plot to the window
p1 = win.addPlot(title=f"Full Waveform {md['data_product']}", axisItems={"left":date_axis})
p1.addItem(i1)
p1.setLabel("left", text="UTC Time")
p1.setLabel("bottom", text="Distance along Fibre", units="m")

# Add a histogram for color scaling
hist = pg.HistogramLUTItem()
hist.setImageItem(i1)
win.addItem(hist)

win.show()

inc = 0
def update_plot():
    global inc

    full, md = client.fetch_data_product(
        list(range(-n_frames + 1, 1, 1)),
        timeout=20000,
    )
    full = full.reshape((full.shape[0] * full.shape[1], -1)).squeeze()

    if full is None:
        return

    # updates image
    i1.setImage(full.transpose(), autoRange=True, autoLevels=inc==0)

    # sets correct time and space axes
    t = np.arange(0, full.shape[0])*md["dT"] + md["acq_times"][0]
    x = np.arange(md['nx']) * md["dx"] + md["sensing_range"][0]
    x = x[x_start:x_stop]

    # Sets plot range so that data is centered on t and x axis.
    dt = np.mean(np.diff(t))
    tmin = t[0] - dt/2
    tmax = t[-1] + dt/2
    xmin = x[0] - md['dx']/2
    xmax = x[-1] + md['dx']/2
    i1.setRect([xmin, tmin, xmax - xmin, tmax - tmin])

    inc += 1

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(int(update_rate * 1000))

if __name__ == "__main__":

    pg.exec()