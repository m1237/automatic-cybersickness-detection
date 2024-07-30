import Dataloader_Sensors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def eye_diameter_changes_plotter(eye_tracking_df):
    y = eye_tracking_df['right_pupil_diameter_value']
    x = list(range(len(eye_tracking_df)))
    # plotting
    plt.title("Eye Diameter Changes")
    plt.xlabel("X axis")
    plt.ylabel("Diameter Values")
    plt.plot(x, y, color="red")
    plt.show()


def blinking_plotter():
    y = pd.read_csv("../data/train/1/20220225-213714_eyetracking_data.csv", usecols=["right_blinking"])
    # y = data['right_blinking']
    print(y.shape)
    print(y)

    y[y == True] = 10
    y[y == False] = 5
    print(y)
    x = list(range(len(eye_tracking_df)))
    # plotting
    plt.title("Blinking Changes")
    plt.xlabel("X axis")
    plt.ylabel("Blinking Values")
    plt.plot(x, y, color="blue")
    plt.ylim(0, 15)
    plt.show()


def eye_tracking_heat_map_sns(eye_tracking_data):
    x = np.array(eye_tracking_data["right_gazeray_direction_x"])
    y = np.array(eye_tracking_data["right_gazeray_direction_y"])
    z = np.array(eye_tracking_data["right_gazeray_direction_z"])
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=25)

    sns.set(rc={'figure.figsize': (15, 15)})
    sns.heatmap(heatmap.T, cbar=False).set(xticklabels=[], yticklabels=[])
    plt.show()

def eye_tracking_heat_map_plt(eye_tracking_data):
    x = np.array(eye_tracking_data["right_gazeray_direction_x"])
    y = np.array(eye_tracking_data["right_gazeray_direction_y"])
    z = np.array(eye_tracking_data["right_gazeray_direction_z"])
    # z = z.reshape((len(x), len(y)))

    #plt.contour(x, y, z, 20, cmap='RdGy');
    #fig.show()

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title('Heatmap for Right Eye')
    plt.show()

    x_left = np.array(eye_tracking_data["left_gazeray_direction_x"])
    y_left = np.array(eye_tracking_data["left_gazeray_direction_y"])

    heatmap_left, xedges_left, yedges_left = np.histogram2d(x_left, y_left, bins=50)
    extent = [xedges_left[0], xedges_left[-1], yedges_left[0], yedges_left[-1]]

    plt.clf()
    plt.imshow(heatmap_left.T, extent=extent, origin='lower')
    plt.title('Heatmap for Left Eye')
    plt.show()


if __name__ == '__main__':

    # CS person
    df = Dataloader_Sensors.get_eyetracking_df('../data/train', '1')
    eye_tracking_heat_map_sns(df)

    # Not CS person
    df = Dataloader_Sensors.get_eyetracking_df('../data/train', '2')
    eye_tracking_heat_map_sns(df)
