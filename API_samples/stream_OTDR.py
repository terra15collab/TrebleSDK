"""
Example script using PYQTGRAPH and Client Functions to plot realtime Terra15 Treble OTDR data.
"""

### Pyside6, Pyside2, or PyQt5 must be installed for correct display.
### Pyside6 is preferred.

### The "treble" API package must be installed to run the Treble API.
### Download links can be provided by Terra15.

### SETUP PARAMETERS ################################################################################
treble_ip = "10.0.0.70"
server_port = "48000"
update_rate = 0.16  # (s)
########################################################################################################

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from treble import acq_client

# setup Treble connection
client = acq_client.acq_Client()
client.connect_to_server(f"tcp://{treble_ip}:{server_port}")
_, md = client.fetch_data_product([-1],timeout=20000)

# CREATE PLOT WINDOW
app = pg.mkQApp("Plotting Example")

win = pg.GraphicsLayoutWidget(title=f"Streaming Treble Data: {treble_ip}")
win.setWindowTitle(f"Streaming Treble Data: {treble_ip}")

p1 = win.addPlot(title="OTDR")
p1.setLabel("bottom", text="Distance along Fibre", units="m")
p1.setLabel("left", text="OTDR")
OTDR_curve = p1.plot(pen="w")

win.show()

inc = 0
def update_plot():
    global inc, p1

    otdr, md = client.fetch_OTDR([-1], timeout=20000)

    if otdr is None:
        return
    otdr = otdr[0][-1]

    # generate full x-vector using dx
    x = np.arange(0, otdr.shape[0]) * md["dx"]
    OTDR_curve.setData(x, otdr)

    inc += 1

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(update_rate * 1000)

if __name__ == "__main__":

    pg.exec()