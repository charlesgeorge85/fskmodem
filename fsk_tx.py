import numpy as np

from utility import load_config
from utility import hex_to_bits
from utility import bits_to_bytes
from utility import baseband_bandlimit
from utility import plot_fsk_psd
from utility import add_awgn

# ============================================================
# SYSTEM PARAMETERS (AS PER YOUR SPEC)
# ============================================================
cfg = load_config("config/config.yaml")
fsk_modem_cfg = cfg["fsk_modem"]

BIT_RATE    = fsk_modem_cfg["bit_rate"]* 1e3
SAMPLE_RATE = fsk_modem_cfg["sampling_rate"]*1e3  # 96 kSps (exact 20 samples/bit)
FREQ_DEV    = fsk_modem_cfg["freq_dev"]*1e3        # ±4.5 kHz
NUM_BITS    = fsk_modem_cfg["num_bits"];
SAMPLES_PER_BIT = int(SAMPLE_RATE / BIT_RATE)  # = 20
TX_NPY_FILE = "tx_fsk.npy"  # OUTPUT INPUT FILE
AWGN_ENABLE = fsk_modem_cfg["awgn_en"]
SNR_DB = fsk_modem_cfg["snr"]
PAD_LEN = fsk_modem_cfg["padding_len"]
CHFIR_ENABLE = fsk_modem_cfg["chfir_en"]
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
padding_bits = np.ones(PAD_LEN, dtype=np.int8)
preamble_bits  = hex_to_bits(0x55_55_55_55_55_55_55_55_93_0B_51_DE, 12*8)  
#preamble_bits  = hex_to_bits(0x7E_7E_7E_7E_7E_7E_7E_7E_7E_7E_7E_7E, 12*8)  
#preamble_bits  = hex_to_bits(0xAA_BF_FF_FF_FC_14_F0_47_21_8E_4F_43, 12*8)  
#preamble_bits  = hex_to_bits(0x82_9E_08_e4, 4*8) # Sending this stream 3 time Gaudian Modem will detect  
#preamble_bits  = hex_to_bits(0x82_9E_08_e8_80_80_00_01_02_03_04_05, 12*8)  
tx_bits = np.random.randint(0, 2, NUM_BITS)
#tx_bits = generate_incremental_bits(1024*10)
#tx_bits = generate_alternating_bits(NUM_BITS)
#tx_bits =generate_repeated_byte_bits(0xF0,20)
#tx_bits = np.ones(NUM_BITS,dtype=np.int8)
#tx_bits = np.tile(preamble_bits, 10)
#tx_bits  = hex_to_bits(0x55_55_55, 3*8)  
postamble_bits = hex_to_bits(0x7E_7E_7E_7E_7E, 5*8)  

tx_bits = np.concatenate([
    padding_bits,
    preamble_bits,
    tx_bits,
    postamble_bits
])

#inp = hex_to_bits(0x7E7E, 16)
#inp = [1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1]
#out = guardian_modem_encoder(inp)
#print("Input :", inp)
#print("Output:", out)
#tx_bits_enc = encode_in_16bit_blocks(tx_bits)

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
# Base Band Filtering
# =========================
if CHFIR_ENABLE:
    tx_iq = baseband_bandlimit(tx_iq, SAMPLE_RATE, 25e3)

# =========================
# SAVE TO .NPY FILE
# =========================
tx_iq = tx_iq.astype(np.complex64)
np.save(TX_NPY_FILE, tx_iq);
np.save("tx_bits.npy", tx_bits);

# ============================================================
# OPTIONAL: VERIFY WAVEFORM (DEBUG / SANITY CHECK)
# ============================================================
# Instantaneous frequency estimation
#phase_diff = np.angle(tx_iq[1:] * np.conj(tx_iq[:-1]))
#inst_freq = phase_diff * SAMPLE_RATE / (2 * np.pi)

num_bytes = 24
num_bits = num_bytes * 8
payload_bits_80 = tx_bits[-num_bits:]
payload_bytes = bits_to_bytes(payload_bits_80)
hex_string = " ".join(f"{b:02X}" for b in payload_bytes)

print("========Generated 2-FSK waveform========")
print(f"Sampling Rate       :{SAMPLE_RATE/1e3} Ksps ")
print(f"Samples per bit     :{SAMPLES_PER_BIT}")
print(f"AWGN                :{AWGN_ENABLE}")
print(f"Channel Filter      :{CHFIR_ENABLE}")
print(f"Samples per bit     :{SAMPLES_PER_BIT}")
print(f"Total Bits          :{len(tx_bits)}")
print(f"First 20 TX bits    :{tx_bits[:20]}");
print(f"Total IQ samples    :{len(tx_iq)}")
print(f"last {num_bytes} tx_bytes    :{hex_string}")

# =========================
# Frequency Spectrum
# =========================
plot_fsk_psd(tx_iq, SAMPLE_RATE)
print("==================================end of FSK Tx===================================================")

