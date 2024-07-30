import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import Dataloader_Sensors
from data_processing.pre_processing_data import ecg_to_heart_rate
from data_visualization.util import json_to_dataframe


def empatica_plotter(user_name, sensor_name):
    # It plots all Empatica's data
    # EXCEPT ACC.JSON

    # Open data
    data_directory = '../data/' + user_name + '/Empatica/' + sensor_name + '.json'
    file_exists = os.path.exists(data_directory)
    if file_exists:
        print('im going to plot data for ' + sensor_name)
        empatica_data = pd.read_json(data_directory)
        data = pd.DataFrame(empatica_data.data.tolist(), index=empatica_data.index)

        # Split data
        np_data = np.array(data)
        value = np_data[:, :1]

        # if sample rate is high
        # take bigger steps
        step = 1
        if len(value) > 300:
            step = 5
        steps = np.arange(start=0, stop=len(value) - 1, step=step)
        selected = value[steps]

        # show on plot
        plt.plot(steps, selected, color='black')
        plt.xlabel('index')
        plt.ylabel(sensor_name)
        plt.show()
    else:
        print("There's no file for " + sensor_name)


def show_empatica_plot_for(user_name):
    sensors = ['EDA', 'IBI', 'TEMP', 'HR_EMPATICA']
    for sensor_name in sensors:
        empatica_plotter(user_name, sensor_name)


def polar_plotter(user_name):
    # Open data
    polar_data = pd.read_json('../data/' + user_name + '/Polar_h10/ECG.json')
    data = pd.DataFrame(polar_data.data.tolist(), index=polar_data.index)

    # Split data
    np_data = np.array(data)
    value = np_data[:, :1]

    # if sample rate is high
    # take bigger steps
    step = 1
    if len(value) > 300:
        step = 50
    steps = np.arange(start=0, stop=len(value) - 1, step=step)
    selected = value[steps]

    # show on plot
    plt.plot(steps, selected, color='black')
    plt.xlabel('index')
    plt.ylabel('ECG')
    plt.show()


def plot(data, name):
    plt.plot(data)
    plt.ylabel(name)
    plt.xlabel('index')
    plt.show()


def plot_before_and_after_preprocessing(data, signal):
    orig = data[signal]
    run = data[signal + '_run']
    pc = data[signal + '_pc']
    min = data[signal + '_min']
    max = data[signal + '_max']

    plt.plot(range(len(orig)), orig, label="BVP")
    # plotting the line 2 points
    plt.plot(range(len(run)), run, label="BVP RUN")
    # plt.plot(range(len(pc)), pc, label="BVP PC")
    plt.plot(range(len(min)), max, label="BVP MIN")
    plt.plot(range(len(min)), max, label="BVP MAX")
    # Set a title of the current axes.
    plt.title('BVP signal and BVP processed features')
    # show a legend on the plot
    plt.legend()

    plt.ylim((0.5, 0.7))
    plt.xlim((0, 25))

    # Display a figure.
    plt.show()


def plot_scatter(x_data, y_data, colors, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(x_data, y_data, c=colors)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)

    fig.show()


def plot_with_cs(data, signal):
    signal = data['hr']
    gt = data['label']

    # we can only look at one dimension at a time
    plot_scatter(range(len(signal)), signal, gt, 'timesteps', 'EDA', 'EDA signal with CS timestamps')


def plot_raw_signal(data):
    plt.plot(range(len(data)), data, label="HR")
    plt.title('HR computed from ECG')
    plt.show()


if __name__ == "__main__":
    user_name = ''
    default_buffer = '0.1 S'
    preprocessed_data = Dataloader_Sensors.get_sensor_data('../data',
                                                           user_name,
                                                           default_buffer)

    columns = ['bvp_run', 'eda_run', 'hr_run', 'ibi_run', 'temp_run', 'ecg_run']
    for column in columns:
        bvp = preprocessed_data[column]
        # plot(bvp, column)

    # MARJA METHOD SECTION
    signal = 'hr'
    default_buffer = '0.5 S'
    buffer_cs = 500000
    filepath = '../data'
    data = Dataloader_Sensors.get_sensor_data('../data', user_name, default_buffer, with_orig=True)
    data_cs = Dataloader_Sensors.add_label(data, Dataloader_Sensors.get_timestamps(filepath, user_name), buffer_cs)
    plot_before_and_after_preprocessing(data, signal)

    bvp_json = os.path.join(filepath, user_name, 'Empatica', 'BVP.json')
    df_bvp = json_to_dataframe(bvp_json, 'bvp')

    hr_json = os.path.join(filepath, user_name, 'Empatica', 'HR_EMPATICA.json')
    df_hr = json_to_dataframe(hr_json, 'hr')

    ecg_json = os.path.join(filepath, user_name, 'Polar_h10', 'ECG.json')
    df_ecg = json_to_dataframe(ecg_json, 'ecg')

    df_hr_computed = ecg_to_heart_rate(df_ecg)

    plot_raw_signal(df_hr['hr'])
    plot_raw_signal(df_hr_computed['hr'])
    plot_with_cs(data_cs, 'eda')
