

import numpy as np
import matplotlib.pyplot as plt
import yaml

def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

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


def hex_to_bits(value, num_bits):
    """
    Convert hex/int value to MSB-first bit array
    """
    return np.array(
        [(value >> (num_bits - 1 - i)) & 1 for i in range(num_bits)],
        dtype=np.int8
    )


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


def generate_repeated_byte_bits(byte_value, repeat_count):
    """
    Generate a bit stream by repeating a byte value.

    Args:
        byte_value (int): Byte value (0x00 to 0xFF)
        repeat_count (int): Number of times to repeat the byte

    Returns:
        np.ndarray (int8): Bit stream (0/1), MSB-first
    """
    if not (0 <= byte_value <= 0xFF):
        raise ValueError("byte_value must be between 0x00 and 0xFF")

    if repeat_count <= 0:
        return np.array([], dtype=np.int8)

    bits = []

    for _ in range(repeat_count):
        for i in range(8):
            bits.append((byte_value >> (7 - i)) & 1)

    return np.array(bits, dtype=np.int8)


# =========================
# TX Filter
# =========================
from scipy.signal import firwin, lfilter

def baseband_bandlimit(iq, fs, bw_hz=25e3):
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

# =========================
# AWGN
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
# =========================
# RX Preamble detection
# =========================
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
# =========================
# Frequency Spectrum
# =========================
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
# =========================
# Power Spectrum over Time
# =========================
def plot_power_time_db(rx_iq, sample_rate):
    power = np.abs(rx_iq) ** 2

    # Avoid log(0)
    power_db = 10 * np.log10(power + 1e-12)

    t = np.arange(len(power_db)) / sample_rate

    plt.figure()
    plt.plot(t, power_db)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Power (dB)")
    plt.title("RX Power vs Time (dB)")
    plt.grid(True)
    plt.show()
