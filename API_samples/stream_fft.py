"""
Example script using PYQTGRAPH and Client Functions to plot realtime Terra15 Treble FFT data.
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
# Crops data between fibre index
x_start = 300
x_stop = 500
########################################################################################################


### DEFAULT CLIENT FUNCTIONS USED TO PROCESS FFT ON TREBLE ############################################
#
# dispatcher = dict()
#
# # Function to average over a range of slices
# def get_slices_by_range_mean(inp, start_index2=None, end_index2=None, extra_md=None):
#     return np.mean(inp[:, :, start_index2:end_index2], axis=2)
#
#
# dispatcher["get_slices_by_range_mean"] = get_slices_by_range_mean
#
#
# # Function to average over a range of slices and take an FFT over the time dimension for all frames
# def get_slices_by_range_mean_rfft_hanning(inp, start_index2=None, end_index2=None, extra_md=None):
#     # Get the mean over a range
#     mn = get_slices_by_range_mean(inp, start_index2, end_index2)
#     avg = np.mean(mn)
#     # Dimension 0 is over indices, dimension 1 is over time, unwrap the array
#     mn_reshape = mn.reshape(mn.size)
#     # avg = np.mean(np.mean(mn_reshape))
#     chunksize = mn_reshape.shape[0]
#     window = np.hanning(chunksize)
#     rfft = np.fft.rfft(mn_reshape * window) / chunksize
#     return rfft
#
# dispatcher["get_slices_by_range_mean_rfft_hanning"] = get_slices_by_range_mean_rfft_hanning

########################################################################################################

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from treble import acq_client

# setup Treble connection
client = acq_client.acq_Client()
client.connect_to_server(f"tcp://{treble_ip}:{server_port}")
_, md = client.fetch_data_product([-1],timeout=20000)

# Calculates x axis to display correct region.
x = np.arange(md['nx']) * md["dx"] + md["sensing_range"][0]

# CREATE PLOT WINDOW
app = pg.mkQApp("Plotting Example")

win = pg.GraphicsLayoutWidget(title=f"Streaming Treble Data: {treble_ip}")
win.setWindowTitle(f"Streaming Treble Data: {treble_ip}")

p1 = win.addPlot(title=f"Avg {md['data_product']} Spectrum between {x[x_start]:.0f}m->{x[x_stop]:.0f}m")
p1.setLabel("bottom", text="Frequency", units="Hz")
p1.setLabel("left", text="fft")
fft_curve = p1.plot(pen="g")

win.show()

inc = 0
def update_plot():
    global inc, p1
    fft, md = client.fetch_data_product(
        list(range(-n_frames + 1, 1, 1)),
        timeout=20000,
        with_client_fn="get_slices_by_range_mean_rfft_hanning",
        client_fn_args={"start_index2": x_start, "end_index2": x_stop}
    )
    if fft is None:
        return

    fft = np.abs(fft)
    freqs = np.fft.rfftfreq(md["nT"]*n_frames, md["dT"])

    fft_curve.setData(freqs, fft)

    inc += 1

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(update_rate * 1000)

if __name__ == "__main__":

    pg.exec()