"""
Example script using PYQTGRAPH and Client Functions to send alerts to Slack based on OTDR thresholds.
"""

### The "treble" API package must be installed to run the Treble API.
### Download links can be provided by Terra15.

### First create a Slack app to post messages to:
### https://api.slack.com/apps


### SETUP PARAMETERS ################################################################################
treble_ip = "10.0.0.139"
server_port = "48000"
t_check = 60 # (s)
otdr_threshold = 0.2
webhook_url = "https://hooks.slack.com/services/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"
########################################################################################################


import numpy as np
from treble import acq_client
import time
from datetime import datetime
import urllib3
import json
import traceback


def send_slack_alert(message):
    try:
        slack_message = {'text': message}
        http = urllib3.PoolManager()
        response = http.request(
            'POST',
            webhook_url,
            body=json.dumps(slack_message),
            headers={'Content-Type': 'application/json'},
            retries=False
        )
    except:
        traceback.print_exc()

    return True


# setup Treble connection
client = acq_client.acq_Client()
client.connect_to_server(f"tcp://{treble_ip}:{server_port}")
_, md = client.fetch_data_product([-1],timeout=20000)

while True:
    check_start = datetime.utcfromtimestamp(time.time())
    msg = f"{check_start}" \
          f"\nOTDR Reflection Check || {md['serial_number']} || {treble_ip} || Threshold = {otdr_threshold}"

    otdr, md = client.fetch_OTDR([-1], timeout=20000)

    if otdr is None:
        continue
    otdr = otdr[0][-1]

    # Generate full x-vector using dx
    x = np.arange(0, otdr.shape[0]) * md["dx"]

    # Locates the start of each reflection point
    ref_loc = np.where(otdr > otdr_threshold)[0]
    ref_loc = ref_loc[np.where(np.diff(ref_loc)>=5)]
    ref_distance = x[ref_loc]

    if len(ref_loc) > 0:
        for dist in ref_distance:
            msg = msg+f"\n   - Reflection @ {dist:.2f}m"

    send_slack_alert(msg)

    # Wait the remainder of the time
    time.sleep(t_check - (time.time() % t_check))