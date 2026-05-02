# This file should not use pandas

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, iirfilter


def notch_filter(data, hz, fs):
    b, a = iirnotch(hz, Q=30, fs=fs)
    y = filtfilt(b, a, data, axis=0)
    return y


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data, axis=0)
    return y


def common_average_montage(ieeg_data):
    """
    Compute the common average montage for iEEG data.

    Parameters:
    - ieeg_data: 2D numpy array
        Rows are data points, columns are electrode channels.

    Returns:
    - cam_data: 2D numpy array
        Data after applying the common average montage.
    """

    # Ensure input is a numpy array
    if not isinstance(ieeg_data, np.ndarray):
        raise ValueError("Input data must be a 2D numpy array.")

    # Ensure the shape of ieeg_data is correct
    if ieeg_data.shape[0] < ieeg_data.shape[1]:
        raise ValueError("ieeg_data must have more rows than columns. ")

    # Compute the average across all channels
    avg_signal = ieeg_data.mean(axis=1)

    # Subtract the average signal from each channel
    result = ieeg_data - avg_signal[:, np.newaxis]

    # Check if the shape of the result matches the shape of ieeg_data
    if result.shape != ieeg_data.shape:
        raise ValueError(
            "The shape of the resulting data doesn't match the input data."
        )

    return result


def electrode_selection(labels):
    """
    Returns label selection array.
    Inputs:
    labels - string array of channel label names
    """
    select = np.ones((len(labels),), dtype=bool)
    for i, label in enumerate(labels):
        label_upper = label.upper()
        # Remove EKG, ECG, RATE, RR
        for check in ["EKG", "ECG", "RATE", "RR"]:
            if check in label_upper:
                select[i] = 0

        # Remove specific scalp channels
        checks = set(
            (
                "C3",
                "C4",
                "CZ",
                "F8",
                "F7",
                "F4",
                "F3",
                "FP2",
                "FP1",
                "FZ",
                "LOC",
                "T4",
                "T5",
                "T3",
                "C6",
                "ROC",
                "P4",
                "P3",
                "T6",
                "O1",  # Remove O1
            )
        )
        if label_upper in checks:
            select[i] = 0

        # Remove any channel that starts with DC
        if label_upper.startswith("DC"):
            select[i] = 0

        # fix for things that could be either scalp or ieeg
        if label_upper == "O2":
            if "O1" in {l.upper() for l in labels}:  # if hemiscalp, should not have odd; if ieeg, should have O1
                select[i] = 1
            else:
                select[i] = 0
    return select


def detect_bad_channels_optimized(values, fs):
    which_chs = np.arange(values.shape[1])

    ## Parameters
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3
    percent_60_hz = 0.1
    mult_std = 10

    bad = set()
    high_ch = []
    nan_ch = []
    zero_ch = []
    high_var_ch = []
    noisy_ch = []

    nans_mask = np.isnan(values)
    zero_mask = values == 0
    nan_count = np.sum(nans_mask, axis=0)
    zero_count = np.sum(zero_mask, axis=0)

    median_values = np.nanmedian(values, axis=0)
    std_values = np.nanstd(values, axis=0)

    median_std = np.nanmedian(std_values)
    higher_std = which_chs[std_values > (mult_std * median_std)]

    for ich in which_chs:
        eeg = values[:, ich]

        # Check NaNs
        if nan_count[ich] > 0.5 * len(eeg):
            bad.add(ich)
            nan_ch.append(ich)
            continue

        # Check zeros
        if zero_count[ich] > (0.5 * len(eeg)):
            bad.add(ich)
            zero_ch.append(ich)
            continue

        # Check above absolute threshold
        if np.sum(np.abs(eeg - median_values[ich]) > abs_thresh) > 10:
            bad.add(ich)
            high_ch.append(ich)
            continue

        # High variance check
        pct = np.percentile(eeg, [100 - tile, tile])
        thresh = [
            median_values[ich] - mult * (median_values[ich] - pct[0]),
            median_values[ich] + mult * (pct[1] - median_values[ich]),
        ]
        if np.sum((eeg > thresh[1]) | (eeg < thresh[0])) >= num_above:
            bad.add(ich)
            high_var_ch.append(ich)
            continue

        # 60 Hz noise check, modified to match original function
        Y = np.fft.fft(eeg - np.nanmean(eeg))
        P = np.abs(Y) ** 2
        freqs = np.linspace(0, fs, len(P) + 1)
        freqs = freqs[:-1]
        P = P[: int(np.ceil(len(P) / 2))]
        freqs = freqs[: int(np.ceil(len(freqs) / 2))]
        total_power = np.sum(P[(freqs > 5)])
        if total_power == 0:
            bad.add(ich)
            high_var_ch.append(ich)
            continue
        else:
            P_60Hz = np.sum(P[(freqs > 58) & (freqs < 62)]) / total_power
            if P_60Hz > percent_60_hz:
                bad.add(ich)
                noisy_ch.append(ich)

    # Combine all bad channels
    bad = bad.union(higher_std)

    details = {
        "noisy": noisy_ch,
        "nans": nan_ch,
        "zeros": zero_ch,
        "var": high_var_ch,
        "higher_std": list(higher_std),
        "high_voltage": high_ch,
    }

    channel_mask = [i for i in which_chs if i not in bad]

    return channel_mask, details


def automatic_bipolar_montage(data, data_columns):
    """This function returns the data in bipolar montage using the channel names.
    
    Handles variable-length channel names by separating letter prefix from numeric suffix.
    Works with formats like: AB01, ROF01, HIPP1, etc.

    Args:
        data (numpy.ndarray): 2D array with shape (timepoints, channels)
        data_columns (list or array): List of channel names

    Returns:
        tuple: (bipolar_data, bipolar_column_names)
            - bipolar_data: numpy array of bipolar montage data
            - bipolar_column_names: array of bipolar channel names (e.g., 'ROF01-ROF02')
    """
    channels = np.array(data_columns)

    nchan = len(channels)
    count = 0
    
    for ch in range(nchan-1):
        ch1Ind = ch
        ch1 = channels[ch1Ind]
        
        # Use regex to separate letter prefix from numeric suffix
        match = re.match(r'([A-Za-z]+)(\d+)', ch1)
        if not match:
            # Skip channels that don't match expected format
            continue
            
        prefix = match.group(1)  # Letter part (e.g., 'ROF', 'AB', 'HIPP')
        number = int(match.group(2))  # Numeric part (e.g., 1, 01, 001)
        num_digits = len(match.group(2))  # Preserve number of digits for formatting
        
        # Find sequential index by incrementing the number
        ch2 = prefix + f"{(number + 1):0{num_digits}d}"

        ch2exists = np.where(channels == ch2)[0]
        if len(ch2exists) > 0:
            ch2Ind = ch2exists[0]
            bipolar = pd.Series((data[:,ch1Ind] - data[:,ch2Ind])).rename(f'{ch1}-{ch2}')
            if count == 0: #initialize
                dfBipolar = pd.DataFrame(bipolar)
                count = count + 1
            else:
                dfBipolar = pd.concat([dfBipolar, pd.DataFrame(bipolar)], axis=1)

    return np.array(dfBipolar), np.array(dfBipolar.columns)

def butter_bp_filter(data, lowcut, highcut, fs, order=3):
    """This function bandpasses data

    Args:
        data (pandas.DataFrame): Pandas dataframe with channels in columns
        lowcut (float): Lower bound of band (Hz)
        highcut (float): Higher bound of band (Hz)
        fs (int): Sample frequency
        order (int, optional): Filter order. Defaults to 3.

    Returns:
        pandas.DataFrame: Filtered data
    """
    bandpass_b, bandpass_a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    signal_bp = filtfilt(bandpass_b, bandpass_a, data, axis=0)
    return signal_bp

def new_bandpass_filt(data, lowcut, highcut, fs, order = 4):

    if fs == 1024:
        b = [0.0342089439540094, 0, -0.0684178879080189, 0, 0.0342089439540094]
        a = [1, -3.40837438183729, 4.36773330454651, -2.50912852732152, 0.549774904492375]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 512:
        b = [0.110341876092030, 0, -0.220683752184059, 0, 0.110341876092030]
        a = [1, -2.84999684509203, 3.01525532511459, -1.47261949410732, 0.307429302286222]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 256:
        b = [0.329773746685091, 0, -0.659547493370182, 0, 0.329773746685091]
        a = [0.175233821173555, -0.200810082151730, 0.830452461803929, -1.80406441093276, 1]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 500:
        b = [0.114657122916782, 0, -0.229314245833564, 0, 0.114657122916782]
        a = [1, -2.82405915548544, 2.95589050957282, -1.43119381095927, 0.299436856070785]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 1000:
        b = [0.0356639677619672, 0, -0.0713279355239343, 0, 0.0356639677619672]
        a = [1, -3.39453926569963, 4.33233599727022, -2.47975692956295, 0.541965991567219]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 250:
        b = [0.342416923871813, 0, -0.684833847743626, 0, 0.342416923871813]
        a = [1, -1.75428158626788, 0.736712673740108, -0.159617332038591, 0.178069768015428]
        signal_bp = filtfilt(b, a, data, axis=0)

    else:
        print("Sampling Frequency is not covered by this function, this is the best approximation")
        print("Spike rates/counts will not be affected by more than 5 percent of the original value")
        bandpass_b, bandpass_a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
        signal_bp = filtfilt(bandpass_b, bandpass_a, data, axis=0)

    return signal_bp

import re

#clean labels
def decompose_labels(chLabel, name):
    """
    clean the channel labels, one at a time.
    """
    clean_label = []
    elec = []
    number = []
    label = chLabel

    if isinstance(label, str):
        label_str = label
    else:
        label_str = label[0]

    # Remove leading zero
    label_num_idx = re.search(r'\d', label_str)
    if label_num_idx:
        label_non_num = label_str[:label_num_idx.start()]
        label_num = label_str[label_num_idx.start():]

        if label_num.startswith('0'):
            label_num = label_num[1:]

        label_str = label_non_num + label_num

    # Remove 'EEG '
    eeg_text = 'EEG '
    if eeg_text in label_str:
        label_str = label_str.replace(eeg_text, '')

    # Remove '-Ref'
    ref_text = '-Ref'
    if ref_text in label_str:
        label_str = label_str.replace(ref_text, '')

    # Remove spaces
    label_str = label_str.replace(' ', '')

    # Remove '-'
    label_str = label_str.replace('-', '')

    # Remove CAR
    label_str = label_str.replace('CAR', '')

    # Switch HIPP to DH, AMY to DA
    label_str = label_str.replace('HIPP', 'DH')
    label_str = label_str.replace('AMY', 'DA')

    # Dumb fixes specific to individual patients
    if name == 'HUP099':
        if label_str.startswith('R'):
            label_str = label_str[1:]

    if name == 'HUP189':
        label_str = label_str.replace('Gr', 'G')

    if name == 'HUP106':
        label_str = label_str.replace('LDA', 'LA')
        label_str = label_str.replace('LDH', 'LH')
        label_str = label_str.replace('RDA', 'RA')
        label_str = label_str.replace('RDH', 'RH')

    if (name == 'HUP086') | (name == 'HUP078'):
        label_str = label_str.replace('Grid', 'LG')

    if name == 'HUP075':
        label_str = label_str.replace('Grid', 'G')

    clean_label = label_str

    if 'Fp1' in label_str.lower():
        clean_label = 'Fp1'

    if 'Fp2' in label_str.lower():
        clean_label = 'Fp2'

    return clean_label