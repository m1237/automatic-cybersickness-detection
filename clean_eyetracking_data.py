import sys
import os
import pandas as pd
import csv
from src.dataloaders.util import json_to_dataframe


def remove_first_row(filepath):
    """ Removes the first row of the eyetracking csv
        Parameters:
            filepath (string): the path to the csv eyetracking file
        Returns:
            None
    """
    lines = list()
    with open(filepath, 'r') as readFile:
        reader = csv.reader(readFile)
        for i, row in enumerate(reader):
            # first row is header, second row is invalid
            if i != 1:
                lines.append(row)

    with open(filepath, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


def calc_new_timestamp(old_timestamp):
    """ Calculates the new time for the timestamp(+1 hour)
        Parameters:
            old_timestamp (string): the timestamp
        Returns:
            new_timestamp (string): the new timestamp (+1 hour)
    """
    timestamp_date = pd.to_datetime(old_timestamp, format="%Y%m%d%H%M%S%f")
    timestamp_date_new = timestamp_date.replace(hour=timestamp_date.hour + 1)
    timestamp_new = timestamp_date_new.strftime(format="%Y%m%d%H%M%S%f")
    return timestamp_new


def adapt_eye_timestamps(filepath):
    """ Sets the eyetracking timestamps to +1 hour
        Parameters:
            filepath (string): the path to the csv eyetracking file
        Returns:
            None
    """
    lines = list()
    with open(filepath, 'r') as readFile:
        reader = csv.reader(readFile)
        for i, row in enumerate(reader):
            # first row is header
            if i != 0:
                # modify timestamp
                old_timestamp = row[0].split(';')[0]
                new_timestamp = calc_new_timestamp(old_timestamp)
                row = [row[0].replace(old_timestamp, new_timestamp)]
                lines.append(row)
            else:
                lines.append(row)

    with open(filepath, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


def adapt_cs_timestamps(filepath):
    """ Sets the cs timestamps to +1 hour
        Parameters:
            filepath (string): the path to the csv cs file
        Returns:
            None
    """
    lines = list()
    with open(filepath, 'r') as readFile:
        reader = csv.reader(readFile)
        for i, row in enumerate(reader):
            # first row is header
            if i != 0:
                # modify timestamp
                old_timestamp = row[0]
                new_timestamp = calc_new_timestamp(old_timestamp)
                row = [row[0].replace(old_timestamp, new_timestamp)]
                lines.append(row)
            else:
                lines.append(row)

    with open(filepath, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


def check_first_row(path):
    """ Iterates over the eyetracking data and check if the first row needs to be removed.
        Parameters:
            path (string): the path to the train/validation data
        Returns:
            None
    """
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            # get eyetracking csv
            for file in os.listdir(os.path.join(path, folder)):
                if 'eyetracking' in file and 'csv' in file:
                    eyetracking_csv = os.path.join(path, folder, file)
                    eyetracking_df = pd.read_csv(eyetracking_csv, sep=';', dtype={'current_timestamp': 'string'})
                    # check if first row needs to be removed
                    if eyetracking_df.iloc[0].timestamp_device == 0:
                        # row is not valid --> remove
                        print(f'Remove first row for participant {folder}')
                        remove_first_row(eyetracking_csv)


def check_eye_timestamp_hours(path):
    """ Checks if the eyetracking timestamps need to be set to +1 hour.
        Parameters:
            path (string): the path to the train/validation data
        Returns:
            None
    """
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            # get eyetracking csv
            for file in os.listdir(os.path.join(path, folder)):
                if 'eyetracking' in file and 'csv' in file:
                    eyetracking_path = os.path.join(path, folder, file)
                    eyetracking_df = pd.read_csv(eyetracking_path, sep=';', dtype={'current_timestamp': 'string'})
            # get sensor file
            sensor_path = os.path.join(path, folder, 'Empatica', 'ACC.json')
            sensor_df = json_to_dataframe(sensor_path, 'empatica_acc_')
            # check if the timestamps are the same
            eye_timestamp = pd.to_datetime(eyetracking_df.iloc[0].current_timestamp, format="%Y%m%d%H%M%S%f")
            sensor_timestamp = pd.to_datetime(sensor_df.iloc[0].timestamp, format="%Y-%m-%d %H:%M:%S.%f")
            if eye_timestamp.time().hour == sensor_timestamp.time().hour:
                # hours are the same --> set eyetracking timestamps to +1 hour
                print(f'Adapting eye timestamps for participant {folder}')
                adapt_eye_timestamps(eyetracking_path)


def check_cs_timestamp_hours(path):
    """ Checks if the cs timestamps need to be set to +1 hour.
        Parameters:
            path (string): the path to the train/validation data
        Returns:
            None
    """
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            # get eyetracking csv
            for file in os.listdir(os.path.join(path, folder)):
                if 'cs_timestamps' in file and 'csv' in file:
                    cs_path = os.path.join(path, folder, file)
                    cs_df = pd.read_csv(cs_path, sep=';', dtype={'current_timestamp': 'string'})
            # get sensor file
            sensor_path = os.path.join(path, folder, 'Empatica', 'ACC.json')
            sensor_df = json_to_dataframe(sensor_path, 'empatica_acc_')
            # check if the timestamps are the same
            cs_timestamp = pd.to_datetime(cs_df.iloc[0].cs_timestamp, format="%Y%m%d%H%M%S%f")
            sensor_timestamp = pd.to_datetime(sensor_df.iloc[0].timestamp, format="%Y-%m-%d %H:%M:%S.%f")
            if cs_timestamp.time().hour == sensor_timestamp.time().hour:
                # hours are the same --> set eyetracking timestamps to +1 hour
                print(f'Adapting cs timestamps for participant {folder}')
                adapt_cs_timestamps(cs_path)



if __name__ == '__main__':
    """ Clean the collected eyetracking csv files.
    For some participants, the first row contains invalid eyetracking data --> remove this row
    For most participants, the hours of the eyetracking timestamps are 1 hour ahead the timestamps of the sensor data.
    But: for some participants, the timestamps are the same --> set the eyetracking timestamps to +1 hour"""

    train_path = os.path.join(sys.path[0], 'Storage', 'training')
    val_path = os.path.join(sys.path[0], 'Storage', 'validation')

    # check if first invalid row needs to be removed
    # check_first_row(train_path)
    # check_first_row(val_path)

    # set eyetracking timestamps to +1 hour
    # check_eye_timestamp_hours(train_path)
    # check_eye_timestamp_hours(val_path)

    # set cs timestamps to +1 hour
    check_cs_timestamp_hours(train_path)
    check_cs_timestamp_hours(val_path)

    sys.exit("All done.")
