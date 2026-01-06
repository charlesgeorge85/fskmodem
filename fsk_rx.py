import numpy as np
import matplotlib.pyplot as plt
import yaml
# ============================================================
# SYSTEM PARAMETERS (MUST MATCH TX)
# ============================================================
def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

cfg = load_config("config/config.yaml")
fsk_modem_cfg = cfg["fsk_modem"]

BIT_RATE    = fsk_modem_cfg["bit_rate"]* 1e3
SAMPLE_RATE = fsk_modem_cfg["sampling_rate"]*1e3  # 96 kSps (exact 20 samples/bit)
FREQ_DEV    = fsk_modem_cfg["freq_dev"]*1e3        # ±4.5 kHz
NUM_BITS    = fsk_modem_cfg["num_bits"];

SAMPLES_PER_BIT = int(SAMPLE_RATE / BIT_RATE)  # = 20

RX_NPY_FILE = "rx_fsk.npy"   # <<< INPUT IQ FILE

def hex_to_bits(value, num_bits):
    """
    Convert hex/int value to MSB-first bit array
    """
    return np.array(
        [(value >> (num_bits - 1 - i)) & 1 for i in range(num_bits)],
        dtype=np.int8
    )

def find_preamble_strict(rx_bits, preamble_bits):
    rx_bits = np.asarray(rx_bits).astype(np.int8)
    preamble_bits = np.asarray(preamble_bits).astype(np.int8)

    L = len(preamble_bits)

    for i in range(len(rx_bits) - L + 1):
    #for i in range(4):
        window = rx_bits[i:i+L]
        #print(f"window = {window},preamble_bits={preamble_bits}");
        if np.array_equal(window, preamble_bits):
            return i

    return None

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


# Known preamble
preamble_bits = hex_to_bits(0xAAAA_AAAA, 32)

# rx_bits comes from FSK demodulator
start = find_preamble_strict(rx_bits, preamble_bits)

if start is None:
    print("Preamble NOT found")
else:
    print(f"Preamble found at bit index {start}")

    # Extract payload (skip preamble)
    payload_start = start + len(preamble_bits)
    payload_bits = rx_bits[payload_start:]


print("Decoded bits:",len(payload_bits))




# ============================================================
# OPTIONAL: DEBUG PLOTS
# ============================================================

def plot_fsk_spectrum(iq, fs, title="FSK RX Spectrum"):
    """
    Plot FFT-based spectrum from complex IQ
    """
    N = len(iq)

    # Windowing (VERY IMPORTANT)
    window = np.hanning(N)
    iq_win = iq * window

    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(iq_win))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

    # Convert to dB
    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(freq/1e3, spectrum_db)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot instantaneous frequency for a few bits
#plt.figure(figsize=(10, 4))
#plt.plot(inst_freq[-1000:])
#plt.axhline(+FREQ_DEV, color='g', linestyle='--')
#plt.axhline(-FREQ_DEV, color='r', linestyle='--')
#plt.title("Instantaneous Frequency (Decoded)")
#plt.ylabel("Frequency (Hz)")
#plt.xlabel("Sample Index")
#plt.grid()
#plt.show()

# ============================================================
# OPTIONAL: BER CHECK (ONLY IF TX BITS ARE KNOWN)
# ============================================================
# If you saved tx_bits separately, load and compare:
tx_bits = np.load("tx_bits.npy")

# Extract payload (skip preamble)
tx_bits = tx_bits[len(preamble_bits):]

ber = np.mean(tx_bits[:len(payload_bits)] != payload_bits[:len(payload_bits)])

#plot_fsk_spectrum(rx_iq, SAMPLE_RATE)


print("Loaded RX IQ samples")
print("Total samples:", len(rx_bits))
print("First 100 TX bits:", tx_bits[:20])
print("First 100 RX bits:", payload_bits[:20])
print("Samples per bit:", SAMPLES_PER_BIT)
print("BER:", ber)
print("==================================end of FSK Rx===================================================")
