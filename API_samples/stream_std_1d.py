"""
Example script using PYQTGRAPH and Client Functions to plot realtime Terra15 Treble Standard Deviation data.
"""

### Pyside6, Pyside2, or PyQt5 must be installed for correct display.
### Pyside6 is preferred.

### The "treble" API package must be installed to run the Treble API.
### Download links are located in the README.md file.

### SETUP PARAMETERS ################################################################################
treble_ip = "10.0.0.70"
server_port = "48000"
n_frames = 1
update_rate = 0.16  # (s)
########################################################################################################


import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from treble import acq_client

# setup Treble connection
client = acq_client.acq_Client()
client.connect_to_server(f"tcp://{treble_ip}:{server_port}")
_, md = client.fetch_data_product([-1], timeout=20000)

# CREATE PLOT WINDOW
app = pg.mkQApp("Plotting Example")

win = pg.GraphicsLayoutWidget(title=f"Streaming Treble Data: {treble_ip}")
win.setWindowTitle(f"Streaming Treble Data: {treble_ip}")

p1 = win.addPlot(title=f"Std. Dev {md['data_product']}")
p1.setLabel("bottom", text="Distance along Fibre", units="m")
p1.setLabel("left", text="Std Dev", units=f"{md['data_product']}")
low_curve = p1.plot(pen="r")
rms_curve = p1.plot(pen="y")

win.show()

low_hold = None
inc = 0
def update_plot():
    global p1, inc, low_hold

    rms, md = client.fetch_data_product(
        list(range(-n_frames + 1, 1, 1)),
        timeout=20000,
        with_client_fn="calc_rms_multiframe",
    )
    if rms is None:
        low_hold = None
        return

    x = np.arange(md["nx"]) * md["dx"] + md["sensing_range"][0]

    rms_curve.setData(x, rms)

    if low_hold is None:
        low_hold = rms

    low_hold = np.minimum(low_hold, rms)

    low_curve.setData(x, low_hold)
    inc += 1

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(update_rate * 1000)

if __name__ == "__main__":

    pg.exec()