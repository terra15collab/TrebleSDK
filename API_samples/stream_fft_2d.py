"""
Example script using PYQTGRAPH and Client Functions to plot full raw waveform Terra15 Treble data.
"""

### Pyside6, Pyside2, or PyQt5 must be installed for correct display.
### Pyside6 is preferred.

### The "treble" API package must be installed to run the Treble API.
### Download links can be provided by Terra15.

### SETUP PARAMETERS ################################################################################
treble_ip = "10.0.0.90"
server_port = "48000"
n_frames = 1
update_rate = 0.16  # (s)
########################################################################################################

import numpy as np
import pyqtgraph as pg
from scipy.signal import welch
from pyqtgraph.Qt import QtCore
from treble import acq_client

# setup Treble connection
client = acq_client.acq_Client()
client.connect_to_server(f"tcp://{treble_ip}:{server_port}")
_, md = client.fetch_data_product([-1],timeout=5000)

# Create window
app = pg.mkQApp("Plotting Example")
win = pg.GraphicsLayoutWidget(title=f"Streaming Treble Data: {treble_ip}")
win.setWindowTitle(f"Streaming Treble Data: {treble_ip}")

# Create image item to update inside plot
i1 = pg.ImageItem()

# Add a plot to the window
p1 = win.addPlot(title=f"Power Spectral Density (dB)")
p1.addItem(i1)
p1.setLabel("left", text="Frequency (Hz)")
p1.setLabel("bottom", text="Distance along Fibre", units="m")

# Add a histogram for color scaling
hist = pg.HistogramLUTItem()
hist.setImageItem(i1)
hist.gradient.loadPreset("inferno")
win.addItem(hist)

win.show()

inc = 0
def update_plot():
    global inc

    full, md = client.fetch_data_product(
        list(range(-n_frames + 1, 1, 1)),
        timeout=1000,
    )
    full = full.reshape((full.shape[0] * full.shape[1], -1)).squeeze()

    f, psd = welch(full, fs=1/md["dT"], nperseg=full.shape[0], axis=0)

    if full is None:
        return

    # updates image
    i1.setImage(10*np.log10(psd.transpose()), autoRange=True, autoLevels=inc==0)

    # sets correct time and space axes
    x = np.arange(md['nx']) * md["dx"] + md["sensing_range"][0]

    # Sets plot range so that data is centered on t and x axis.
    fmin=f[0]
    fmax=f[-1]
    xmin = x[0] - md['dx']/2
    xmax = x[-1] + md['dx']/2
    i1.setRect([xmin, fmin, xmax - xmin, fmax-fmin])

    inc += 1

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(update_rate * 1000)

if __name__ == "__main__":

    pg.exec()