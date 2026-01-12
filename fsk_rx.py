import numpy as np
from utility import load_config
from utility import hex_to_bits
from utility import bits_to_bytes
from utility import find_preamble_strict
from utility import plot_fsk_psd
from utility import plot_power_time_db
from rx_analyse import detect_active_bits

# ============================================================
# SYSTEM PARAMETERS (MUST MATCH TX)
# ============================================================
cfg = load_config("config/config.yaml")
fsk_modem_cfg = cfg["fsk_modem"]

BIT_RATE    = fsk_modem_cfg["bit_rate"]* 1e3
SAMPLE_RATE = fsk_modem_cfg["sampling_rate"]*1e3  # 96 kSps (exact 20 samples/bit)
FREQ_DEV    = fsk_modem_cfg["freq_dev"]*1e3        # ±4.5 kHz
NUM_BITS    = fsk_modem_cfg["num_bits"];
SAMPLES_PER_BIT = int(SAMPLE_RATE / BIT_RATE)  # = 20
RX_NPY_FILE = "rx_fsk.npy"   # <<< INPUT IQ FILE
PAD_LEN = fsk_modem_cfg["padding_len"]
LPBK = fsk_modem_cfg["loopBack_en"]
# ============================================================
# LOAD RX IQ SAMPLES
# ============================================================
rx_iq = np.load(RX_NPY_FILE).astype(np.complex64)
# ============================================================
# FSK DEMODULATION (QUADRATURE DISCRIMINATOR)
# ============================================================

# Phase difference between consecutive samples
phase_diff = np.angle(rx_iq[1:] * np.conj(rx_iq[:-1]))

# Convert phase change to instantaneous frequency
inst_freq = phase_diff * SAMPLE_RATE / (2 * np.pi)

# Remove DC bias (important even in simulation)
inst_freq -= np.mean(inst_freq)

# ============================================================
# BIT DECISION
# ============================================================
rx_bits = []

num_rx_bits = len(inst_freq) // SAMPLES_PER_BIT

for i in range(num_rx_bits):
    start = i * SAMPLES_PER_BIT
    stop  = start + SAMPLES_PER_BIT

    freq_segment = inst_freq[start:stop]

    # Decision: sign of average frequency
    bit = 1 if np.mean(freq_segment) > 0 else 0
    rx_bits.append(bit)

rx_bits = np.array(rx_bits, dtype=np.int8)

# ============================================================
# Detect Active Data
# ============================================================

iq_idx, bit_idx, power_db = detect_active_bits(
    rx_iq,
    rx_bits,
    SAMPLE_RATE,
    SAMPLES_PER_BIT,
    -10 #power_threshold_db
)
print("First active IQ index:", iq_idx[0])
print("First active power index:", power_db[0])
print("First active bit index:", bit_idx[0])

rx_bits = rx_bits[bit_idx[0]:]

# ============================================================
# Known preamble
# ============================================================
#preamble_bits  = hex_to_bits(0x55_55_55_55_55_55_55_55, 8*8)  
preamble_bits  = hex_to_bits(0x55_55_55_55_55_55_55_55_93_0B_51_DE, 12*8)  

# ============================================================
# rx_bits comes from FSK demodulator
# ============================================================
start = find_preamble_strict(rx_bits, preamble_bits)

if start is None:
    print("\033[31mPreamble NOT found\033[0m")
else:
    print(f"\033[32mPreamble found at bit index {start}\033[0m")

# Extract payload (skip preamble)
payload_start = start + len(preamble_bits)
payload_bits = rx_bits[payload_start:]

print("Decoded bits:",len(payload_bits))

# ============================================================
# OPTIONAL: BER CHECK (ONLY IF TX BITS ARE KNOWN)
# ============================================================
if LPBK:

    # If you saved tx_bits separately, load and compare:
    tx_bits = np.load("tx_bits.npy")

    # Extract payload (skip preamble)
    tx_bits = tx_bits[PAD_LEN + len(preamble_bits):]

    dataBitsLength = len(tx_bits)-8;
    print(f"tx_bits : {len(tx_bits)}; rx_bits : {len(payload_bits)}")
    ber = np.mean(tx_bits[:dataBitsLength] != payload_bits[:dataBitsLength])

    payload_bits = payload_bits[:dataBitsLength+8]

    num_bytes = 24
    num_bits = num_bytes * 8
    payload_bits_80 = payload_bits[-num_bits:]
    payload_bytes = bits_to_bytes(payload_bits_80)
    hex_string = " ".join(f"{b:02X}" for b in payload_bytes)

print("========Recieved 2-FSK waveform========")
print(f"Sampling Rate        :{SAMPLE_RATE/1e3} Ksps ")
print(f"Samples per bit      :{SAMPLES_PER_BIT}")
print(f"Total samples        :{len(rx_bits)}")
print(f"First 20 RX bits     :{payload_bits[:20]}")
if LPBK:
    print(f"First 20 TX bits     :{tx_bits[:20]}")
    print(f"BER                  :{ber}")
    print(f"last {num_bytes} rx_bytes    :{hex_string}")
plot_fsk_psd(rx_iq, SAMPLE_RATE)
plot_power_time_db(rx_iq,SAMPLE_RATE)
print("==================================end of FSK Rx===================================================")
