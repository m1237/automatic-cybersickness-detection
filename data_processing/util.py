import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


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


def acc_json_to_dataframe(json_file_acc):
    with open(json_file_acc) as json_f:
        data = json.load(json_f)
        output = list()
        for row in data['data'].items():
            value_list = row[1][0]
            x_value = value_list[0]
            y_value = value_list[1]
            z_value = value_list[2]
            timestamp = row[1][1]  # TODO: not correct, but needed?
            values = {'x_value': x_value, 'y_value': y_value, 'z_value': z_value, 'timestamp': timestamp}
            output.append(values)
    return pd.DataFrame(output)


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    txt_file = 'test.txt'
    PATH = '../Polar_h10/ECG.json'
    values = json_to_np_array(json_file=PATH)
    json_to_txt(PATH, txt_file)
    FILE_PATH_ACC = '../Polar_h10/ACC.json'
    acc_json_to_dataframe(FILE_PATH_ACC)
