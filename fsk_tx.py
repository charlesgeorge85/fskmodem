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

def generate_incremental_bits(num_bytes):
    """
    Generate a bitstream for FSK transmission where payload bytes
    increment from 0x00 to 0xFF and repeat.

    Args:
        num_bytes (int): Number of payload bytes

    Returns:
        np.ndarray (int8): Bitstream (0/1), MSB-first
    """
    if num_bytes <= 0:
        return np.array([], dtype=np.int8)

    # Generate incremental bytes 00..FF repeating
    byte_pattern = np.arange(256, dtype=np.uint8)
    repeats = (num_bytes + 255) // 256
    bytes_out = np.tile(byte_pattern, repeats)[:num_bytes]

    # Convert bytes to bits (MSB-first)
    bits = []
    for b in bytes_out:
        bits.extend([(b >> (7 - i)) & 1 for i in range(8)])

    return np.array(bits, dtype=np.int8)



def generate_alternating_bits(length, start_bit=1):
    """
    Generate an alternating bit sequence: 1010... or 0101...

    Args:
        length (int): Number of bits to generate
        start_bit (int): 1 for 1010..., 0 for 0101...

    Returns:
        np.ndarray (int8): Bit sequence (0/1)
    """
    if length <= 0:
        return np.array([], dtype=np.int8)

    start_bit = 1 if start_bit else 0
    bits = (np.arange(length) + start_bit) % 2

    return bits.astype(np.int8)




np.random.seed(0)
padding_bits = np.ones(PAD_LEN, dtype=np.int8)
preamble_bits  = hex_to_bits(0x55_55_55_55_55_55_55_55_93_0B_51_DE, 12*8)  
#preamble_bits  = hex_to_bits(0x7E_7E_7E_7E_7E_7E_7E_7E_7E_7E_7E_7E, 12*8)  
#tx_bits = np.random.randint(0, 2, NUM_BITS)
#tx_bits = generate_incremental_bits(1024*10)
#tx_bits = generate_alternating_bits(NUM_BITS)
tx_bits = np.ones(NUM_BITS,dtype=np.int8)
postamble_bits = hex_to_bits(0xAA_AA_AA_AA_AA, 5*8)  

tx_bits = np.concatenate([
    #padding_bits,
    #preamble_bits,
    tx_bits,
    #postamble_bits
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


# =========================
# TX Filter
# =========================
from scipy.signal import firwin, lfilter

def apply_tx_bandlimit(iq, fs, bw_hz=25e3):
    """
    Apply digital low-pass filter to limit TX bandwidth
    """
    cutoff = bw_hz / 2
    taps = firwin(
        numtaps=129,
        cutoff=cutoff,
        fs=fs
    )
    return lfilter(taps, 1.0, iq)




tx_iq_filtered = apply_tx_bandlimit(tx_iq, SAMPLE_RATE, 20e3)
#tx_iq_filtered = tx_iq

print("Generated 2-FSK waveform")
print("Samples per bit :", SAMPLES_PER_BIT)
print("Total Bits      :", len(tx_bits))
#print("First 100 TX bits:", tx_bits[:100]);
print("Total IQ samples:", len(tx_iq))

############################
# SAVE TO .NPY FILE
############################
tx_iq = tx_iq.astype(np.complex64)
np.save(TX_NPY_FILE, tx_iq_filtered);
np.save("tx_bits.npy", tx_bits);

# ============================================================
# OPTIONAL: VERIFY WAVEFORM (DEBUG / SANITY CHECK)
# ============================================================
# Instantaneous frequency estimation
phase_diff = np.angle(tx_iq[1:] * np.conj(tx_iq[:-1]))
inst_freq = phase_diff * SAMPLE_RATE / (2 * np.pi)


def plot_fsk_psd(iq, fs, title="FSK Power Spectral Density"):
    N = len(iq)

    window = np.hanning(N)
    iq_win = iq * window

    spectrum = np.fft.fftshift(np.fft.fft(iq_win))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

    # PSD normalization
    psd = (np.abs(spectrum)**2) / (np.sum(window**2) * fs)
    psd_db = 10 * np.log10(psd + 1e-15)

    plt.figure(figsize=(10, 4))
    plt.plot(freq / 1e3, psd_db)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_fsk_psd(tx_iq_filtered, SAMPLE_RATE)


def bits_to_bytes(bits):
    """
    Convert array of bits (0/1) to byte array
    Assumes MSB-first bit order
    """
    bits = np.asarray(bits).astype(np.uint8)
    bytes_out = []

    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | b
        bytes_out.append(byte)

    return np.array(bytes_out, dtype=np.uint8)

num_bytes = 256
num_bits = num_bytes * 8
payload_bits_80 = tx_bits[:num_bits]
payload_bytes = bits_to_bytes(payload_bits_80)
hex_string = " ".join(f"{b:02X}" for b in payload_bytes)
print(f"last {num_bytes} tx_bytes:", hex_string)






# Plot instantaneous frequency for first few bits
#plt.figure(figsize=(10, 4))
#plt.plot(inst_freq[-100:])
#plt.title("Instantaneous Frequency (First Few Bits)")
#plt.ylabel("Frequency (Hz)")
#plt.xlabel("Sample Index")
#plt.grid()
#plt.show()
print("==================================end of FSK Tx===================================================")

