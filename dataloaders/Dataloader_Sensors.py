import os
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable
from torch.utils.data import Dataset

from dataloaders.pre_processing_data import ecg_to_heart_rate, hr_to_ibi
from dataloaders.util import acc_json_to_dataframe, json_to_dataframe
from dataloaders.util import merge_dfs, fill_milliseconds, substitute_bools, generate_timeseries_data


def preprocess_acc(df_acc):
    """
    Preprocess Acceleration values from the Empatica sensor.
    Create relative values to previous row.
    :param df_acc: DataFrame containing acceleration coordinates from Empatica sensor
    :return: DataFrame of acceleration values where each row contains relative value to previous row
    """
    # create new columns for calculation
    df_acc['empatica_acc_x_value_progr'] = 0
    df_acc['empatica_acc_y_value_progr'] = 0
    df_acc['empatica_acc_z_value_progr'] = 0
    # iterate over DataFrame for calculation of new values
    for idx in range(1, len(df_acc)):
        df_acc.loc[idx, 'empatica_acc_x_value_progr'] = df_acc.loc[idx, 'empatica_acc_x_value'] - df_acc.loc[
            idx - 1, 'empatica_acc_x_value']
        df_acc.loc[idx, 'empatica_acc_y_value_progr'] = df_acc.loc[idx, 'empatica_acc_y_value'] - df_acc.loc[
            idx - 1, 'empatica_acc_y_value']
        df_acc.loc[idx, 'empatica_acc_z_value_progr'] = df_acc.loc[idx, 'empatica_acc_z_value'] - df_acc.loc[
            idx - 1, 'empatica_acc_z_value']
    return df_acc.drop(columns=['empatica_acc_x_value', 'empatica_acc_y_value', 'empatica_acc_z_value'])


def scale(data: pd.Series, scaler: str):
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'standard':
        scaler = StandardScaler()
    return scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()


def preprocess_sensor_data(sensors, dfs, buffer,
                                             df_acc, df_h10_acc, df_bvp, df_eda, df_hr, df_ibi, df_temp, df_ecg):
    """Capsuled in another function for live prediction"""

    for df in dfs:
        df['timestamp'] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
        # 2 hours for evaluation (summer time), 1 hour for training
        df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=2)  # bc of different system times

    # scale all sensor values
    for df, col in zip(dfs, sensors):
        df[col] = scale(df[col], 'minmax')
    df_acc['empatica_acc_y_value'] = scale(df_acc['empatica_acc_y_value'], 'minmax')
    df_acc['empatica_acc_z_value'] = scale(df_acc['empatica_acc_z_value'], 'minmax')
    df_h10_acc['h10_acc_y_value'] = scale(df_h10_acc['h10_acc_y_value'], 'minmax')
    df_h10_acc['h10_acc_z_value'] = scale(df_h10_acc['h10_acc_z_value'], 'minmax')

    # any other preprocessing (relative values, etc.)
    # acc = preprocess_acc(df_acc)  # relative values

    # compute rolling moving average with window of n frames
    # possible other moving averages: expanding, exponential (https://towardsdatascience.com/moving-averages-in-python-16170e20f6c)
    n = 10
    for df, col in zip(dfs, sensors):
        df[col + "_run"] = df[col].rolling(n, min_periods=1).mean()
    df_acc['empatica_acc_y_value_run'] = df_acc['empatica_acc_y_value'].rolling(n, min_periods=1).mean()
    df_acc['empatica_acc_z_value_run'] = df_acc['empatica_acc_z_value'].rolling(n, min_periods=1).mean()
    df_h10_acc['h10_acc_y_value_run'] = df_h10_acc['h10_acc_y_value'].rolling(n, min_periods=1).mean()
    df_h10_acc['h10_acc_z_value_run'] = df_h10_acc['h10_acc_z_value'].rolling(n, min_periods=1).mean()

    # compute percentage of change for n periods
    for df, col in zip(dfs, sensors):
        df[col + "_pc"] = df[col].pct_change(periods=n)
    df_acc['empatica_acc_y_value_pc'] = df_acc['empatica_acc_y_value'].pct_change(
        periods=n)
    df_acc['empatica_acc_z_value_pc'] = df_acc['empatica_acc_z_value'].pct_change(periods=n)
    df_h10_acc['h10_acc_y_value_pc'] = df_h10_acc['h10_acc_y_value'].pct_change(periods=n)
    df_h10_acc['h10_acc_z_value_pc'] = df_h10_acc['h10_acc_z_value'].pct_change(periods=n)

    # compute max and min value for n frames
    for df, col in zip(dfs, sensors):
        df[col + "_max"] = df[col].rolling(n, min_periods=1).max()
        df[col + "_min"] = df[col].rolling(n, min_periods=1).min()
    df_acc['empatica_acc_y_value_max'] = df_acc['empatica_acc_y_value'].rolling(n, min_periods=1).max()
    df_acc['empatica_acc_y_value_min'] = df_acc['empatica_acc_y_value'].rolling(n, min_periods=1).min()
    df_acc['empatica_acc_z_value_max'] = df_acc['empatica_acc_z_value'].rolling(n, min_periods=1).max()
    df_acc['empatica_acc_z_value_min'] = df_acc['empatica_acc_z_value'].rolling(n, min_periods=1).min()
    df_h10_acc['h10_acc_y_value_max'] = df_h10_acc['h10_acc_y_value'].rolling(n, min_periods=1).max()
    df_h10_acc['h10_acc_y_value_min'] = df_h10_acc['h10_acc_y_value'].rolling(n, min_periods=1).min()
    df_h10_acc['h10_acc_z_value_max'] = df_h10_acc['h10_acc_z_value'].rolling(n, min_periods=1).max()
    df_h10_acc['h10_acc_z_value_min'] = df_h10_acc['h10_acc_z_value'].rolling(n, min_periods=1).min()

    # merge to one big dataframe
    # df_eyetracking_with_ground_truth['timestamp'] += df_acc['timestamp'][20] - \
    #                                                 df_eyetracking_with_ground_truth['timestamp'][20]

    df: pd.DataFrame = merge_dfs([df_acc, df_bvp, df_eda, df_hr, df_ibi, df_temp, df_h10_acc, df_ecg], buffer)
    # df: pd.DataFrame = merge_dfs([df_bvp, df_eda, df_hr, df_ibi, df_temp, df_ecg], buffer)
    df.sort_index(inplace=True)
    # df['timestamp'] = df.index
    df.reset_index(inplace=True)

    pc_keys = [key for key in df.keys() if "_pc" in key]
    df[pc_keys] = df[pc_keys].replace(np.nan, 0)
    for key in pc_keys:
        max = df[key].replace(np.inf, 0).max()
        min = df[key].replace(-np.inf, 0).min()
        df[key] = df[key].replace(np.inf, max)
        df[key] = df[key].replace(-np.inf, min)

    # drop all original values
    sensors.extend(['empatica_acc_y_value', 'empatica_acc_z_value', 'h10_acc_y_value', 'h10_acc_z_value'])
    df.drop(columns=sensors, inplace=True)
    # print(df)
    df.interpolate(inplace=True)  # interpolate missing values
    df.dropna(inplace=True)
    return df  # ==> one row == timestamp pro 0.5sec, number of cols - 1 == input layer size, last col == y


def get_sensor_data(path_training_data, name, buffer):
    sensors = ['empatica_acc_x_value', 'bvp', 'eda', 'hr', 'ibi', 'temp', 'h10_acc_x_value', 'ecg']
    # for one folder (person)
    try:
        acc_json = os.path.join(path_training_data, name, 'Empatica', 'ACC.json')
        bvp_json = os.path.join(path_training_data, name, 'Empatica', 'BVP.json')
        eda_json = os.path.join(path_training_data, name, 'Empatica', 'EDA.json')
        hr_json = os.path.join(path_training_data, name, 'Empatica', 'HR_EMPATICA.json')
        ibi_json = os.path.join(path_training_data, name, 'Empatica', 'IBI.json')
        temp_json = os.path.join(path_training_data, name, 'Empatica', 'TEMP.json')
        h10_acc_json = os.path.join(path_training_data, name, 'Polar_h10', 'ACC.json')
        ecg_json = os.path.join(path_training_data, name, 'Polar_h10', 'ECG.json')
    except FileNotFoundError as exception:
        print(f'Could not find Sensor data. Skipping user {name}')
        raise exception

    # dataframe for each file of every sensor
    df_acc = acc_json_to_dataframe(acc_json, 'empatica_acc_')  # relative to previous row
    df_bvp = json_to_dataframe(bvp_json, 'bvp')  # scale
    df_eda = json_to_dataframe(eda_json, 'eda')  # leave
    df_temp = json_to_dataframe(temp_json, 'temp')  # same formular as hr
    df_h10_acc = acc_json_to_dataframe(h10_acc_json, 'h10_acc_')  # relative to previous row
    df_ecg = json_to_dataframe(ecg_json, 'ecg')
    if os.path.exists(hr_json):
        df_hr = json_to_dataframe(hr_json, 'hr')  # compute formular from paper
    else:
        df_hr = ecg_to_heart_rate(df_ecg)  # compute hr using biosppy package
    if os.path.exists(ibi_json):
        df_ibi = json_to_dataframe(ibi_json, 'ibi')  # leave
    else:
        df_ibi = hr_to_ibi(df_hr)
    dfs = [df_acc, df_bvp, df_eda, df_hr, df_ibi, df_temp, df_h10_acc, df_ecg]

    # 2. Preprocessing
    # capsuled in a function for live prediction
    preprocessed_df = preprocess_sensor_data(sensors, dfs, buffer,
                                             df_acc, df_h10_acc, df_bvp, df_eda, df_hr, df_ibi, df_temp, df_ecg)
    return preprocessed_df


def get_sensor_data_for_all_persons_by_folder(path):
    """ Loads the sensor data for all participants.
        Parameters:
            path (string): the path to the sensor data
        Returns:
            df (dataframe): the sensor dataframe
    """
    df = pd.DataFrame()
    for person in os.listdir(path):
        if os.path.isdir(os.path.join(path, person)):
            df = pd.concat([df, get_sensor_data(path, person)])
    return df


def get_timestamps(filepath, person):
    """ Loads the cs timestamp data for one participant.
        Parameters:
            filepath (string): the path to the cs timestamp data
            person (string): the name of the participant's folder
        Returns:
            label_df (dataframe): the dataframe containing the cs timestamps
    """
    for file in os.listdir(os.path.join(filepath, person)):
        if 'cs_timestamps' in file and 'csv' in file:
            cs_timestamps_csv = os.path.join(filepath, person, file)
            break
    # open cs timestamps as dataframe
    labels = pd.read_csv(cs_timestamps_csv, sep=';', dtype={'cs_timestamp': 'string'})['cs_timestamp'].to_numpy()

    labels_formatted = []
    for label in labels:
        label_formatted = fill_milliseconds(label)
        labels_formatted.append(label_formatted)

    # create new label df with start and end time created using the buffer
    label_df = pd.DataFrame(labels_formatted, columns=['original_label'])
    return label_df


def add_artificial_label(df, factor):
    df.loc[df['label'] > 0, 'label'] = 1
    df['label'] = df['label'].fillna(0)
    pre_post_values = factor
    saved_indices = []
    for idx, row in df.iloc[pre_post_values:].iterrows():
        pre_rows = df.loc[:idx].tail(pre_post_values)
        post_rows = df.loc[idx:].head(pre_post_values)
        # add label 1 before label 1
        if row['label'] == 1 and pre_rows.iloc[-2]['label'] == 0:
            saved_indices.extend(list(pre_rows.index.values))
        # add label 1 after last occurring label 1
        if row['label'] == 1 and post_rows.iloc[1]['label'] == 0:
            saved_indices.extend(list(post_rows.index.values))
    df.loc[saved_indices, 'label'] = 1
    return df


def add_label(user_df_merged, label_df, buffer):
    """ Merges the dataframe containing sensor/eyetracking data with the cs timestamps.
        Parameters:
            user_df_merged (dataframe): dataframe containing the sensor/eyetracking data
            label_df (dataframe): dataframe containing the cs timestamps
            buffer (int): timedelta for merging
        Returns:
            user_df_merged (dataframe): the dataframe containing the sensor/eyetracking data with cs timestamps
    """
    # create new label df with start and end time created using the buffer
    buffer = pd.Timedelta(microseconds=buffer)
    label_df = pd.DataFrame(label_df, columns=['original_label'])
    label_df['original_label'] = pd.to_datetime(label_df['original_label'], format="%Y%m%d%H%M%S%f")
    label_df['start'] = label_df['original_label'] - buffer
    label_df['end'] = label_df['original_label'] + buffer

    user_df_merged['label'] = 0

    # iterate over each entry from the labels file, select the corresponding frames and label them accordingly:
    for idx, cs_timestamp in label_df.iterrows():
        annotated_df = (user_df_merged.index >= cs_timestamp["start"]) & (user_df_merged.index <= cs_timestamp["end"])
        user_df_merged.loc[annotated_df, "label"] = 1  # set label to 1 if cs occured
    return user_df_merged


def get_eyetracking_df(filepath, folder):
    """ Loads the eyetracking data for one participant.
        Parameters:
            filepath (string): the path to the eyetracking data
            folder (string): the name of the participant's folder
        Returns:
            eyetracking_df (dataframe): the eyetracking dataframe (selected features, standardized)
    """
    # open csv file as dataframe
    for file in os.listdir(os.path.join(filepath, folder)):
        if 'eyetracking' in file and 'csv' in file:
            eyetracking_csv = os.path.join(filepath, folder, file)
            break
    eyetracking_df = pd.read_csv(eyetracking_csv, sep=';', dtype={'current_timestamp': 'string'})
    eyetracking_df = preprocess_eyetracking_data(eyetracking_df)
    return eyetracking_df


def preprocess_eyetracking_data(eyetracking_df):
    """ Preprocesses the respective measurement data.
    Parameters:
        eyetracking_df (dataframe): the dataframe containing the complete eyetracking data
    Returns:
        features (dataframe): the eyetracking dataframes (selected features, standardized)
    """
    # get features
    fields = ['left_pupil_diameter_value', 'right_pupil_diameter_value',
                'left_gazeray_direction_x', 'left_gazeray_direction_y', 'left_gazeray_direction_z',
                'right_gazeray_direction_x', 'right_gazeray_direction_y', 'right_gazeray_direction_z']
    features = eyetracking_df[fields]

    # substitute bool values in df with 0/1
    features = substitute_bools(features)

    # scale features
    # z = (x - u) / s (u=mean, s=standard deviation)
    scaler = StandardScaler()
    features[fields] = scaler.fit_transform(features[fields])

    features['timestamp'] = pd.to_datetime(eyetracking_df["current_timestamp"], format="%Y%m%d%H%M%S%f")

    return features


class SensorData(Dataset):
    """Loads and merges the data from the different sensors."""

    def __init__(self, filepath, use_eyetracking_data, use_sensor_data, buffer, buffer_cs_timestamps,
                 overwrite, timesteps, algorithm):
        """
            Args:
                filepath: path to the folder where the data is located
                use_eyetracking_data: if eyetracking data is loaded
                use_sensor_data: if sensor data is loaded
                buffer: buffer for merging the dataframes of the data
                buffer_cs_timestamps: buffer for merging the eyetracking/sensor with the cs timesteps
                overwrite: if already created pickle files should be overwritten
                timesteps: number of timesteps that are processed at once by the LSTM
                algorithm: the algorithm that is used (svm/lstm/cnn)
        """
        self.filepath = filepath
        self.use_eyetracking_data = use_eyetracking_data
        self.use_sensor_data = use_sensor_data
        self.buffer = buffer
        self.buffer_cs_timestamps = buffer_cs_timestamps
        self.overwrite = overwrite
        self.timesteps = timesteps
        self.algorithm = algorithm

        file_name = 'sensor_data.pkl'

        if overwrite or file_name not in os.listdir(filepath):
            # merge eyetracking, sensor data and cs timestamps
            df_list_all_users = list()
            for i, person in enumerate(tqdm(os.listdir(filepath))):
                if os.path.isdir(os.path.join(self.filepath, person)):
                    user_dfs = []
                    if use_sensor_data:
                        print("Loading sensor data...")
                        user_dfs.append(get_sensor_data(self.filepath, person, self.buffer))
                    if use_eyetracking_data:
                        print("Loading eyetracking data...")
                        user_dfs.append(get_eyetracking_df(self.filepath, person))
                    self.label_df = get_timestamps(filepath, person)
                    user_df_merged = merge_dfs(user_dfs, self.buffer)
                    user_df_merged = add_label(user_df_merged, self.label_df, self.buffer_cs_timestamps)
                    user_df_merged.dropna(inplace=True)
                    # user_df_merged = add_artificial_label(user_df_merged, 5)
                    df_list_all_users.append(user_df_merged)
            self.df_all_users = pd.concat(df_list_all_users)
            pickle.dump(self.df_all_users, open(os.path.join(filepath, file_name), "wb"))
        else:
            with open(os.path.join(filepath, file_name), "rb") as data:
                self.df_all_users = pickle.load(data)

        x = self.df_all_users.drop(columns=['label']).values
        y = self.df_all_users['label'].values

        # use variables so that gradients can be computed
        x = Variable(torch.tensor(x, dtype=torch.float32))
        y = Variable(torch.tensor(y, dtype=torch.float32))

        if self.algorithm != 'svm':
            # reshape the features to shape (n_rows, 1, n_features)
            x = x.unsqueeze(1)
            # reshape the labels to shape (n_rows, 1)
            y = y.unsqueeze(1)
            # generate timeseries data (of shape n_rows, seq_len, n_features
            x = generate_timeseries_data(x, self.timesteps)

        self.features = x
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]


if __name__ == '__main__':
    # get_sensor_data('../Storage/training', 'michi')
    df = get_sensor_data_for_all_persons_by_folder(f'../data/train')
    print(df.shape)
