import re
import pandas as pd


def read_params_lstm(config_path):
    """ Reads and returns the model parameters from the config txt file (for LSTM model).
        Parameters:
            config_path (src): The path to the config txt file

        Returns:
            num_classes (int): The number of output classes
            input_size (int): The feature size for the LSTM
            hidden_size1 (int): The LSTM hidden size
            hidden_size2 (int): The dense NN hidden size
            dropout_lstm (float): The dropout probability for the LSTM layer
            dropout_p: The dropout probability for the dropout layer
        """
    with open(config_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '(lstm)' in line:
                regex = re.match('^.+LSTM\((\d+), (\d+).+dropout=(\d+.\d+).+$', line)
                input_size = regex.group(1)
                hidden_size1 = regex.group(2)
                dropout_lstm = regex.group(3)
            elif '(dropout)' in line:
                regex = re.match('^.+Dropout\(p=(\d+.\d+).+$', line)
                dropout_p = regex.group(1)
            elif '(fc)' in line:
                regex = re.match('^.+in_features=(\d+).+out_features=(\d+).+$', line)
                hidden_size2 = regex.group(1)
                num_classes = regex.group(2)

    return int(num_classes), int(input_size), int(hidden_size1), int(hidden_size2), float(dropout_lstm), float(dropout_p)


def read_params_cnn(config_path):
    """ Reads and returns the model parameters from the config txt file (for CNN model).
        Parameters:
            config_path (src): The path to the config txt file

        Returns:
            num_classes (int): The number of output classes
            feature_size (int): The feature size for the CNN
            input_size (int): The feature size for the LSTM
            hidden_size (int): The LSTM hidden size
            hidden_size2 (int): The dense NN hidden size
            dropout_lstm (float): The dropout probability for the LSTM layer
            dropout_p: The dropout probability for the dropout layer
        """
    with open(config_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '(conv1)' in line:
                regex = re.match('^.+Conv1d\((\d+).+$', line)
                feature_size = regex.group(1)
            elif '(lstm)' in line:
                regex = re.match('^.+LSTM\((\d+), (\d+).+dropout=(\d+.\d+).+$', line)
                input_size = regex.group(1)
                hidden_size1 = regex.group(2)
                dropout_lstm = regex.group(3)
            elif '(dropout)' in line:
                regex = re.match('^.+Dropout\(p=(\d+.\d+).+$', line)
                dropout_p = regex.group(1)
            elif '(fc)' in line:
                regex = re.match('^.+in_features=(\d+).+out_features=(\d+).+$', line)
                hidden_size2 = regex.group(1)
                num_classes = regex.group(2)

    return int(num_classes), int(feature_size), int(input_size), int(hidden_size1), int(hidden_size2), float(dropout_lstm), float(dropout_p)


def get_data_last_x_seconds(df, timestamp_column_name, timestamp_format, time_now, time_last_x_seconds):
    """ Reads and returns the model parameters from the config txt file (for CNN model).
        Parameters:
            df (Dataframe): dataframe containing all data
            timestamp_column_name (string): name of column in the dataframe where the timestamp is
            timestamp_format (string): format of the timestamp (e.g. "%Y%m%d%H%M%S%f")
            time_now (datetime): end time
            time_last_x_seconds (datetime): start time

        Returns:
            df (Dataframe): The dataframe with data from the las x seconds
    """
    # add timestamps as index to get data from last x seconds
    df['temp_timestamp'] = pd.to_datetime(df[timestamp_column_name], format=timestamp_format)
    data_df = df.set_index(['temp_timestamp'])
    # get data from last x seconds
    data_last_x_seconds_df = data_df.between_time(time_last_x_seconds.time(), time_now.time())
    data_last_x_seconds_df.reset_index()
    return data_last_x_seconds_df
