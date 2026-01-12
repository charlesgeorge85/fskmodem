import uhd
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

############################
# User Parameters
############################
TX_NPY_FILE = "tx_fsk.npy"  # <<< INPUT FILE
RX_NPY_FILE = "rx_fsk.npy"  # >>> OUTPUT FILE


def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

cfg = load_config("config/config.yaml")
usrp_cfg = cfg["usrp"]

CENTER_FREQ =  usrp_cfg["center_freq"]*1e6  
SAMPLE_RATE =  usrp_cfg["sampling_rate"]*1e3
TX_GAIN = -25              # dB
RX_GAIN = 0                # dB
RX_DURATION_SEC = 2           # Capture duration (seconds)
NUM_SAMPLES = int(SAMPLE_RATE * RX_DURATION_SEC)
#===========================
# Load TX samples from .npy
#===========================
tx_signal = np.load(TX_NPY_FILE).astype(np.complex64)
num_samples = len(tx_signal)

print("Loaded TX samples :", num_samples)

#===========================
# Create USRP Object
#===========================
usrp = uhd.usrp.MultiUSRP("master_clock_rate=184.32e6")
#usrp = uhd.usrp.MultiUSRP("master_clock_rate=200e6")

usrp.set_clock_source("external")   # or "gpsdo"
usrp.set_time_source("external")

usrp.set_tx_rate(SAMPLE_RATE,0)
usrp.set_rx_rate(SAMPLE_RATE,0)

usrp.set_tx_freq(CENTER_FREQ,0)
usrp.set_rx_freq(CENTER_FREQ,0)

usrp.set_tx_gain(TX_GAIN,0)
usrp.set_rx_gain(RX_GAIN,0)

usrp.set_tx_antenna("TX/RX",0)
usrp.set_rx_antenna("RX2",0)

time.sleep(1)

#===========================
# TX Streamer
#===========================
tx_streamer = usrp.get_tx_stream(
    uhd.usrp.StreamArgs("fc32", "sc16")
)

tx_md = uhd.types.TXMetadata()
tx_md.start_of_burst = True
tx_md.end_of_burst = False

#===========================
# RX Streamer
#===========================
rx_streamer = usrp.get_rx_stream(
    uhd.usrp.StreamArgs("fc32", "sc16")
)

rx_md = uhd.types.RXMetadata()
rx_buffer = np.zeros(NUM_SAMPLES, dtype=np.complex64)

#===========================
# START RX FIRST (IMPORTANT)
#===========================
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
usrp.issue_stream_cmd(stream_cmd)

#print("RX started...")
#usrp.issue_stream_cmd(
#    uhd.types.StreamCMD(
#        uhd.types.StreamMode.start_cont
#    )
#)
#
time.sleep(0.05)  # allow RX pipeline to arm

#===========================
# TRANSMIT FROM .NPY
#===========================
num_sent = tx_streamer.send(tx_signal, tx_md)
print("TX samples sent :", num_sent)

tx_md.start_of_burst = False
tx_md.end_of_burst = True
tx_streamer.send(np.zeros(1, dtype=np.complex64), tx_md)

#===========================
# RECEIVE
#===========================
#num_rx = rx_streamer.recv(rx_buffer, rx_md, timeout=3.0)
num_rx = rx_streamer.recv(rx_buffer, rx_md, timeout=RX_DURATION_SEC + 1.0)

#===========================
# STOP RX
#===========================
usrp.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
rx_buffer = rx_buffer[:num_rx]
print(f"RX samples received : {num_rx}")

#usrp.issue_stream_cmd(
#    uhd.types.StreamCMD(
#        uhd.types.StreamMode.stop_cont
#    )
#)
#
#rx_buffer = rx_buffer[:num_rx]
rx_buffer = rx_buffer.astype(np.complex64)
#===========================
# SAVE TO .NPY FILE
#===========================
np.save(RX_NPY_FILE, rx_buffer)

#===========================
# Power Calculations
#===========================
def compute_power_db(samples):
    return 10 * np.log10(np.mean(np.abs(samples)**2) + 1e-12)

tx_power_db = compute_power_db(tx_signal)
rx_power_db = compute_power_db(rx_buffer)

############################
# Plot Time Domain
############################
#plt.figure(figsize=(12, 6))
#
#plt.subplot(2, 1, 1)
#plt.plot(np.real(tx_signal[-200:]))
#plt.title("Transmitted Signal (from .npy)")
#plt.xlabel("Samples")
#plt.ylabel("Amplitude")
#
#plt.subplot(2, 1, 2)
#plt.plot(np.real(rx_buffer[-200:]))
#plt.title("Received Signal (Loopback)")
#plt.xlabel("Samples")
#plt.ylabel("Amplitude")
#
#plt.tight_layout()
#plt.show()

#===========================
# Print Results
#===========================
print("========== POWER MEASUREMENTS ==========")
print(f"Center Freq : {usrp.get_tx_freq()/1e6:.2f} Mhz")
print(f"Sampling Rate : {usrp.get_tx_rate()/1e3:.2f} Ksps")
print(f"TX Gain : {usrp.get_tx_gain():.2f} dB (relative)")
print(f"RX Gain : {usrp.get_rx_gain():.2f} dB (relative)")
print(f"TX Baseband Power : {tx_power_db:.2f} dB (relative)")
print(f"RX Baseband Power : {rx_power_db:.2f} dB (relative)")
print(f"Path Loss (TX-RX) : {tx_power_db - rx_power_db:.2f} dB")

print("\033[33m======================================end of usrp loopback====================================\033[0m")
