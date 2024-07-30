import json

import numpy as np
import pandas as pd
import torch


def json_to_txt(json_file, txt_file):
    with open(json_file) as json_f:
        data = json.load(json_f)
        output = list()
        for row in data['data'].items():
            values = row[1][0]
            output.append(values)
        with open(txt_file, "w") as textfile:
            for idx, element in enumerate(output):
                textfile.write(str(element) + "\n")


def json_to_np_array(json_file):
    with open(json_file) as json_f:
        data = json.load(json_f)
        output = list()
        for row in data['data'].items():
            values = row[1][0]
            output.append(values)
    return np.array(output)


def acc_json_to_dataframe(json_file_acc, prefix=''):
    with open(json_file_acc) as json_f:
        data = json.load(json_f)
        output = list()
        for row in data['data'].items():
            value_list = row[1][0]
            x_value = value_list[0]
            y_value = value_list[1]
            z_value = value_list[2]
            timestamp = row[1][1]  # not correct
            values = {f'{prefix}x_value': x_value, f'{prefix}y_value': y_value, f'{prefix}z_value': z_value,
                      'timestamp': timestamp}
            output.append(values)
    return pd.DataFrame(output)


def json_to_dataframe(json_file, value_name='value'):
    with open(json_file) as json_f:
        data = json.load(json_f)
        output = list()
        for row in data['data'].items():
            values = row[1][0]
            timestamp = row[1][1]
            output.append({value_name: values, 'timestamp': timestamp})
    return pd.DataFrame(output)


def acc_json_to_dataframe_live(data, prefix=''):
    output = list()
    for row in data.items():
        value_list = row[1][0]
        x_value = value_list[0]
        y_value = value_list[1]
        z_value = value_list[2]
        timestamp = row[1][1]  # not correct
        values = {f'{prefix}x_value': x_value, f'{prefix}y_value': y_value, f'{prefix}z_value': z_value,
                      'timestamp': timestamp}
        output.append(values)
    return pd.DataFrame(output)


def json_to_dataframe_live(data, value_name='value'):
    output = list()
    for row in data.items():
        values = row[1][0]
        timestamp = row[1][1]
        output.append({value_name: values, 'timestamp': timestamp})
    return pd.DataFrame(output)


def substitute_bools(dataframe):
    """ Substitutes all boolean values in the dataframe with 0/1.
            Parameters:
                dataframe (dataframe): The dataframe containing boolean values

            Returns:
                dataframe: The dataframe with substituted boolean values
    """
    dataframe = dataframe.replace(True, 1)
    dataframe = dataframe.replace(False, 0)
    return dataframe


def fill_milliseconds(timestamp):
    """ Fills the milliseconds of the timestamp up so that they have 6 chars (necessary for datetime conversion.
            Parameters:
                timestamp (str): The timestamp from the csv file

            Returns:
                str: The timestamp with 6-digit-milliseconds
    """

    while len(timestamp) < 17:
        timestamp += "0"
    return timestamp


def fix_timestamps(timestamps):
    """
    Every approx. 0.5 seconds the devices received 73 samples, but they were all mapped to one timestamp.
    Fix problem by ...
    :param timestamps: Series of timestamps
    :return: fixed Series of timestamps
    """
    pass  # TODO


def merge_dfs(dfs, time_span):
    """
    Get 2 dfs and merge them into one df according to their timestamps.
    :param dfs: list of dfs to be merged
    :return: one merged df
    """
    dataframe = None
    for idx, df in enumerate(dfs):
        df['timestamp'] = df['timestamp'].dt.round(time_span)
        grouped_df = df.groupby('timestamp').mean()
        if idx == 0:
            dataframe = grouped_df
        else:
            dataframe = dataframe.merge(grouped_df, how='outer', left_on='timestamp', right_on='timestamp')
    return dataframe


def generate_timeseries_data(features, timesteps):
    """ Generates timeseries data.
        Parameters:
            features (Tensor): Tensor containing the features (n_rows, 1, n_features)
            timesteps (int): The number of timesteps that the NN processes at once (= sequence length)
        Returns:
            features (Tensor): Tensor containing the features plus last n timesteps (n_rows, n_timesteps, n_features
    """

    timeseries_tensor = torch.empty(features.shape[0], timesteps, features.shape[2])
    for row in range(timesteps-1, features.shape[0]):
        timeseries_tensor[row] = torch.cat([features[(row-timesteps+1):row].squeeze(1), features[row]])
    return timeseries_tensor