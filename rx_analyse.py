import numpy as np

def detect_power_indices(rx_iq, power_threshold_db):
    # Instantaneous power
    power = np.abs(rx_iq) ** 2
    power_db = 10 * np.log10(power + 1e-12)

    # Indices where power exceeds threshold
    active_indices = np.where(power_db > power_threshold_db)[0]

    return active_indices, power_db

def iq_indices_to_bit_indices(iq_indices, sps, num_bits):
    # Convert sample index → symbol/bit index
    bit_indices = (iq_indices // sps).astype(int)

    # Keep valid bit indices only
    bit_indices = bit_indices[bit_indices < num_bits]

    # Remove duplicates
    bit_indices = np.unique(bit_indices)

    return bit_indices

def detect_active_bits(rx_iq, rx_bits, Fs, sps, power_threshold_db):
    # Step 1: Detect IQ power threshold crossings
    iq_indices, power_db = detect_power_indices(rx_iq, power_threshold_db)

    # Step 2: Map IQ indices to bit indices
    bit_indices = iq_indices_to_bit_indices(
        iq_indices,
        sps,
        len(rx_bits)
    )

    return iq_indices, bit_indices, power_db

