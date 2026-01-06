import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SYSTEM PARAMETERS (MUST MATCH TX)
# ============================================================
BIT_RATE    = 4.8e3        # 4.8 kbps
SAMPLE_RATE = 96e3         # 96 kSps
FREQ_DEV    = 4.5e3        # ±4.5 kHz
NUM_BITS    = 1000

SAMPLES_PER_BIT = int(SAMPLE_RATE / BIT_RATE)  # = 20

RX_NPY_FILE = "tx_fsk.npy"   # <<< INPUT IQ FILE

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

print("Decoded bits:", len(rx_bits))

# ============================================================
# OPTIONAL: DEBUG PLOTS
# ============================================================

print("Loaded RX IQ samples")
print("Total samples:", len(rx_iq))
print("First 100 RX bits:", rx_bits[-20:]);
print("Samples per bit:", SAMPLES_PER_BIT)



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
ber = np.mean(tx_bits[:len(rx_bits)] != rx_bits)
print("BER:", ber)
print("==================================end of FSK Tx===================================================")
