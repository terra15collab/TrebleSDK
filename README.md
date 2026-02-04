
# Terra15 TrebleSDK

TrebleSDK Provides a library of examples of how to interact with the Terra15 Treble Distributed Acoustic Sensor (DAS).

Sample code shows how to process data from .hdf5 data files and realtime from a Treble server.

## Get Started
### Running Python scripts
1. Install a package manager for python. We recommend [uv](https://docs.astral.sh/uv/) for ease of use and speed, but any package manager will work.
    - If [uv](https://docs.astral.sh/uv/) is not already installed, it can be downloaded and installed using the instructions here:
https://docs.astral.sh/uv/#installation  
    - [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) is another widely used alternative


3. Create a python virtual environment and install the required dependencies.

    On Linux
   ```
   uv venv --python=3.10
   source .venv/bin/activate
   wget --content-disposition https://terra15.com.au/download/latestlinuxapi_cp310_v6.whl
   uv pip install treble-*-linux_x86_64.whl
   uv pip install -r requirements.txt
   sudo apt install x11-apps libgl1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libsm6
   ```

     On Windows (Powershell)  
    ```
    uv venv --python=3.10
    .\.venv\Scripts\activate.ps1
    Invoke-WebRequest -Uri "https://terra15.com.au/download/latestwindowsapi_cp310_v6.whl" -OutFile "."
    uv pip install (Get-ChildItem treble-*-win_amd64.whl | Select-Object -First 1).FullName
    uv pip install -r requirements.txt
    ```

    Download links for the Treble API package are also available here:
    - Linux: https://terra15.com.au/download/latestlinuxapi_cp310_v6.whl
    - Windows: https://terra15.com.au/download/latestwindowsapi_cp310_v6.whl
    - MacOS: https://terra15.com.au/download/latestmacosapi_cp310_v6.whl

3. The SDK provides examples to stream and plot data from a running Treble server. 
   - First edit the script to point to a Treble IP address, eg. `10.0.0.70`.
   - Then run the script as below:
    ```
    uv run API_samples/stream_time_series.py
    ```

4. The SDK also contains processing and plotting examples for Treble .hdf5 data.

    ```
    uv run HDF5_samples/plot_hdf5_sdev.py
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
