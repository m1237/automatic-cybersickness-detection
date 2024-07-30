import numpy as np
import pandas as pd
import pylab as pl
from biosppy import storage
from biosppy.signals import ecg
import neurokit2 as nk
import heartpy as hp
from matplotlib import pyplot as plt
from scipy.signal import resample

from util import json_to_np_array, acc_json_to_dataframe


def biospy():
    signal, mdata = storage.load_txt('test.txt')
    Fs = mdata['sampling_rate']
    N = len(signal)  # number of samples
    T = (N - 1) / Fs  # duration
    ts = np.linspace(0, T, N, endpoint=False)  # relative timestamps
    pl.plot(ts, signal, lw=2)
    pl.grid()
    pl.show()

    out = ecg.ecg(signal=signal, sampling_rate=Fs, show=True)


def neurokit(file_path):
    # Generate 15 seconds of ECG signal (recorded at 250 samples / second)
    ecg_test = nk.ecg_simulate(duration=15, sampling_rate=250, heart_rate=70)
    ecg = json_to_np_array(file_path)
    # Process it
    signals, info = nk.ecg_process(ecg, sampling_rate=130)

    # Visualise the processing
    fig = nk.ecg_plot(signals, sampling_rate=250)
    fig.show()


def heartpy(file_path):
    ecg_data = json_to_np_array(file_path)[:500]
    sample_rate = 130
    plt.figure(figsize=(12, 4))
    plt.plot(ecg_data)
    # plt.show()

    # and zoom in a bit
    plt.figure(figsize=(12, 4))
    plt.plot(ecg_data[0:1000])
    plt.show()

    # run analysis
    wd, m = hp.process(ecg_data, sample_rate)

    # visualise in plot of custom size
    hp.plotter(wd, m, figsize=(12, 4))

    # display computed measures
    for measure in m.keys():
        print('%s: %f' % (measure, m[measure]))

    filtered = hp.filter_signal(ecg_data, cutoff=0.05, sample_rate=sample_rate, filtertype='notch')

    # visualize again
    plt.figure(figsize=(12, 4))
    plt.plot(filtered)
    plt.show()

    # and zoom in a bit
    plt.figure(figsize=(12, 4))
    plt.plot(ecg_data[0:2500], label='original signal')
    plt.plot(filtered[0:2500], alpha=0.5, label='filtered signal')
    plt.legend()
    plt.show()

    # run analysis
    wd, m = hp.process(hp.scale_data(filtered), sample_rate)

    # visualise in plot of custom size
    hp.plotter(wd, m, figsize=(12, 4), title='Filtered Peak Detection HR')

    # display computed measures
    # for measure in m.keys():
    # print('%s: %f' % (measure, m[measure]))

    # resample the data. Usually 2, 4, or 6 times is enough depending on original sampling rate
    resampled_data = resample(filtered, len(filtered) * 2)

    # And run the analysis again. Don't forget to up the sample rate as well!
    wd, m = hp.process(hp.scale_data(resampled_data), sample_rate * 2)

    # visualise in plot of custom size
    plt.figure(figsize=(12, 4))
    hp.plotter(wd, m, figsize=(12, 4), title='HR Peak Detection Resampled Data')

    # display computed measures
    for measure in m.keys():
        print('%s: %f' % (measure, m[measure]))

    hp.plot_poincare(wd, m)

    # print poincare measures
    poincare_measures = ['sd1', 'sd2', 's', 'sd1/sd2']
    print('\nnonlinear poincare measures:')
    for measure in poincare_measures:
        print('%s: %f' % (measure, m[measure]))

    # Output for Neural Net Input
    # TODO: what type of data needs to be output for neural net


def ecg_to_hr(file_path):
    sample_rate = 130
    ecg_data = json_to_np_array(file_path)[:500]
    filtered = hp.filter_signal(ecg_data, cutoff=0.05, sample_rate=sample_rate, filtertype='notch')
    # resample the data. Usually 2, 4, or 6 times is enough depending on original sampling rate
    resampled_data = resample(filtered, len(filtered) * 2)
    # And run the analysis again. Don't forget to up the sample rate as well!
    wd, m = hp.process(hp.scale_data(resampled_data), sample_rate * 2)
    print(m)

def preprocess_acc(file_path):
    df = acc_json_to_dataframe(file_path)
    fig, ax = plt.subplots()
    ax.plot(df['x_value'], label='x_value')
    ax.plot(df['y_value'], label='y_value')
    ax.plot(df['z_value'], label='z_value')
    ax.set_title('Acceleration Values')
    ax.legend()
    fig.show()

def ecg_to_heart_rate(ecg_df):
    ecg_signal = ecg.ecg(ecg_df['ecg'], sampling_rate=130, show=False)
    hr_df = pd.DataFrame()
    ecg_df["timestamp"] = pd.to_datetime(ecg_df['timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
    last = ecg_df['timestamp'].iloc[-1]
    first = ecg_df['timestamp'].iloc[0]
    diff = last - first
    hr_df['hr'] = ecg_signal['heart_rate']
    timestamps = [first]
    hr_length = hr_df['hr'].shape[0] - 1
    for i in range(hr_length):
        timestamps.append(timestamps[-1] + (diff / hr_length))
    hr_df['timestamp'] = timestamps
    return hr_df

def hr_to_ibi(hr_df):
    df = pd.DataFrame()
    df['timestamp'] = hr_df['timestamp']
    df['ibi'] = 60 / hr_df['hr']
    return df


if __name__ == '__main__':
    # ECG
    FILE_PATH_ECG = '../Polar_h10/ECG.json'
    # neurokit(FILE_PATH_ECG)
    heartpy(FILE_PATH_ECG)
    # biospy()

    # ACC
    FILE_PATH_ACC = '../Empatica/ACC.json'
    # preprocess_acc(FILE_PATH_ACC)