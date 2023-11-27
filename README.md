
# Terra15 TrebleSDK

TrebleSDK Provides a library of examples of how to interact with the Terra15 Treble Distributed Acoustic Sensor (DAS).

Sample code shows how to process data from both .hdf5 data files and realtime from a Treble server.

### Running Python scripts
1. If Miniconda is not already installed, it can be downloaded and installed as per instructions below:
   - Linux: https://conda.io/projects/conda/en/stable/user-guide/install/linux.html
   - Windows: https://conda.io/projects/conda/en/stable/user-guide/install/windows.html

2. Create a new conda environment and install package requirements
     
   ```
   conda create --name treblesdk python=3.9
   conda activate treblesdk
   wget --content-disposition https://terra15.com.au/download/latestlinuxapi_cp39_v6.whl
   pip install treble-*-linux_x86_64.whl
   conda install pyqt pyqtgraph
   sudo apt install x11-apps libgl1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libsm6
   ```

3.  Download and Install the current Treble Client API. API links for different platforms are below:
   - Linux: https://terra15.com.au/download/latestlinuxapi_cp39_v6.whl
   - Windows: https://terra15.com.au/download/latestwindowsapi_cp39_v6.whl
   - MacOS: https://terra15.com.au/download/latestmacosapi_cp39_v6.whl


4. The SDK provides examples to stream and plot data from a running Treble server. 
   - First edit the script to point to a Treble IP address, eg. `10.0.0.70`.
   - Then run the script as below:
    ```
    python API_samples/stream_time_series.py
    ```

5. The SDK also contains processing and plotting examples for Treble .hdf5 data.

    ```
    python HDF5_samples/plot_hdf5_sdev.py
    ```


## Contents

**/HDF5_samples/**
- Examples of how to read Treble hdf5 files.

**/API_samples/** 
- Examples of how to load + plot data in realtime from a Treble Server.

**/MATLAB Samples/** 
- Octave/Matlab .m file equivalents to HDF5_samples.

**/sample_data/**
- Small example datasets for testing scripts on real Treble data.