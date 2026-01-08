import numpy as np
import matplotlib.pyplot as plt
import yaml

# ============================================================
# SYSTEM PARAMETERS (AS PER YOUR SPEC)
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

TX_NPY_FILE = "tx_fsk.npy"  # OUTPUT INPUT FILE

AWGN_ENABLE = fsk_modem_cfg["awgn_enable"]
SNR_DB = fsk_modem_cfg["snr"]
PAD_LEN = fsk_modem_cfg["padding_len"]

# =========================
# UTILITIES
# =========================
def add_awgn(signal, snr_db):
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(len(signal)) +
        1j*np.random.randn(len(signal))
    )
    return signal + noise
def hex_to_bits(value, num_bits):
    """
    Convert hex/int value to MSB-first bit array
    """
    return np.array(
        [(value >> (num_bits - 1 - i)) & 1 for i in range(num_bits)],
        dtype=np.int8
    )


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
padding_bits = np.zeros(PAD_LEN, dtype=np.int8)
preamble_bits  = hex_to_bits(0x55_55_55_55_55_55_55_55_93_0B_51_DE, 12*8)  
tx_bits = np.random.randint(0, 2, NUM_BITS)
postamble_bits = hex_to_bits(0xAA_AA, 2*8)  

tx_bits = np.concatenate([
    padding_bits,
    preamble_bits,
    tx_bits,
    postamble_bits
])

tx_iq = bfsk_modulate(
    tx_bits,
    SAMPLE_RATE,
    SAMPLES_PER_BIT,
    FREQ_DEV
)


# =========================
# AWGN
# =========================
if AWGN_ENABLE:
    tx_iq = add_awgn(tx_iq, SNR_DB)

print("Generated 2-FSK waveform")
print("Samples per bit :", SAMPLES_PER_BIT)
print("Total Bits      :", len(tx_bits))
print("First 100 TX bits:", tx_bits[:100]);
print("Total IQ samples:", len(tx_iq))

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

