"""
Example script using PYQTGRAPH to replay data from a Terra15 Treble .hdf5 data file.
"""

### Set playback parameters ###############################################################
file_path = "../sample_data/example_triggered_shot.hdf5"

t_offset = 0
t_window = 1
data_step_rate = 0.05
update_rate = 0.1
##########################################################################################


import numpy as np
import h5py
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

def convert_velocity_to_strainrate(data, gauge_length_m, dx):
    gauge_samples = int(round(gauge_length_m / dx))
    return (data[:, gauge_samples:] - data[:, :-gauge_samples]) / (gauge_samples * dx)


def correct_gauge_length_offset(x, gauge_length):
    """Compensate for distance shift of data caused by gauge_length calculation."""
    # crops end of x_vector by gauge length
    dx = x[1] - x[0]
    gauge_samples = int(round(gauge_length / dx))
    gauge_length = gauge_samples * dx
    x_correct = x[:-gauge_samples]

    # compensates for GL/2 signal offset
    x_correct = x_correct + gauge_length / 2
    return x_correct


app = pg.mkQApp("Plotting Example")

date_axis = pg.DateAxisItem(orientation="bottom", utcOffset=0)
win = pg.PlotWidget(axisItems={"bottom":date_axis})
pg.setConfigOptions(antialias=True)
i1 = pg.ImageItem()
win.addItem(i1)
win.show()
win.setWindowTitle(f"Example Data: {file_path}")
win.setLabel("bottom", text="UTC Time")
win.setLabel("left", text="Distance", units="m")


inc = 0
def update_data():
    global i1, inc
    with h5py.File(file_path, "r") as f:
        data_group = f["data_product"]
        md = dict(f.attrs)
        dt = md["dt_computer"]
        # print(md)
        t1 = int(t_offset/dt) + int((inc * data_step_rate) / dt)
        t2 = t1 + int(t_window / dt)

        # loads data
        data = f["data_product"]["data"][t1:t2]
        t = data_group["posix_time"][t1:t2]
        x = np.arange(md["nx"]) * md["dx"] + md["sensing_range_start"]

        data = convert_velocity_to_strainrate(
            data, md["gauge_length"], md["dx"]
        )
        x = correct_gauge_length_offset(x, md["gauge_length"])

    i1.setImage(data, autoLevels=inc==0)
    i1.setColorMap("viridis")

    i1.setRect([t[0], x[0], t[-1] - t[0], x[-1] - x[0]])

    # steps window forward by update interval
    if t2 > md["nt"]:
        inc = 0
    else:
        inc+=1


timer = QtCore.QTimer()
timer.timeout.connect(update_data)
timer.start(int(update_rate * 1000))

if __name__ == "__main__":
    pg.exec()
