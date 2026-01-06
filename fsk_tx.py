import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SYSTEM PARAMETERS (AS PER YOUR SPEC)
# ============================================================
BIT_RATE    = 4.8e3        # 4.8 kbps
SAMPLE_RATE = 96e3         # 96 kSps (exact 20 samples/bit)
FREQ_DEV    = 4.5e3        # ±4.5 kHz
NUM_BITS    = 1000

SAMPLES_PER_BIT = int(SAMPLE_RATE / BIT_RATE)  # = 20

TX_NPY_FILE = "tx_fsk.npy"  # <<< INPUT FILE

# ============================================================
# 2-FSK (CPFSK) MODULATOR
# ============================================================
def bfsk_modulate(bits, fs, spb, freq_dev):
    """
    Continuous-phase 2-FSK modulator
    bits     : array of 0/1
    fs       : sample rate (Hz)
    spb      : samples per bit (integer)
    freq_dev : frequency deviation (Hz)
    """

    bits = np.asarray(bits).astype(np.int8)

    # NRZ mapping: 0 -> -1, 1 -> +1
    symbols = 2 * bits - 1

    # Upsample symbols
    symbols_upsampled = np.repeat(symbols, spb)

    # Instantaneous frequency
    inst_freq = freq_dev * symbols_upsampled

    # Phase accumulator (continuous phase)
    phase = np.cumsum(2 * np.pi * inst_freq / fs)
    
    # Complex baseband IQ
    iq = np.exp(1j * phase)

    return iq.astype(np.complex64)

# ============================================================
# TEST SIGNAL GENERATION
# ============================================================
np.random.seed(0)
tx_bits = np.random.randint(0, 2, NUM_BITS)

tx_iq = bfsk_modulate(
    tx_bits,
    SAMPLE_RATE,
    SAMPLES_PER_BIT,
    FREQ_DEV
)

print("Generated 2-FSK waveform")
print("Samples per bit:", SAMPLES_PER_BIT)
print("First 100 TX bits:", tx_bits[-20:]);
print("Total samples:", len(tx_iq))

############################
# SAVE TO .NPY FILE
############################
tx_iq = tx_iq.astype(np.complex64)
np.save(TX_NPY_FILE, tx_iq);
np.save("tx_bits.npy", tx_bits);

# ============================================================
# OPTIONAL: VERIFY WAVEFORM (DEBUG / SANITY CHECK)
# ============================================================
# Instantaneous frequency estimation
phase_diff = np.angle(tx_iq[1:] * np.conj(tx_iq[:-1]))
inst_freq = phase_diff * SAMPLE_RATE / (2 * np.pi)

# Plot instantaneous frequency for first few bits
#plt.figure(figsize=(10, 4))
#plt.plot(inst_freq[-100:])
#plt.title("Instantaneous Frequency (First Few Bits)")
#plt.ylabel("Frequency (Hz)")
#plt.xlabel("Sample Index")
#plt.grid()
#plt.show()
print("==================================end of FSK Tx===================================================")

