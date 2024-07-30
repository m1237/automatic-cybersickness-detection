import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


if __name__ == '__main__':
    """ Main method.
        Reads the result files from the model training process and visualizes them as plots.
        Saves the plots to disk.

    Parameters:
        None

    Returns:
        None

    """

    print(f"Reading loss files...")

    loss_files = [os.path.join(sys.path[0], '20220324-230759_51_lstm_0.783_0.071', 'lstm_losses.csv'),
                    os.path.join(sys.path[0], '20220325-165424_256_cnn_0.652_0.000', 'cnn_losses.csv')]

    # create separate plots for all train/validation losses
    fig, ax = plt.subplots(nrows=len(loss_files), ncols=2)

    for row_idx, row in enumerate(ax):
        for col_idx, col in enumerate(row):
            loss_df = pd.read_csv(loss_files[row_idx], sep=';')
            if col_idx == 0:
                # train
                loss_type = 'Train Loss'
            else:
                # validation
                loss_type = 'Validation Loss'
            ax[row_idx, col_idx].set_ylim([0, 0.6])
            ax[row_idx, col_idx].plot(np.arange(len(loss_df[loss_type]))*5, loss_df[loss_type])
            ax[row_idx, col_idx].set_xlabel('Epochs')
            if row_idx == 0 and col_idx == 0:
                ax[row_idx, col_idx].set_ylabel('LSTM')
                ax[row_idx, col_idx].set_title(loss_type)
            elif row_idx == 1 and col_idx == 0:
                ax[row_idx, col_idx].set_ylabel('CNN-LSTM')
            elif row_idx == 0 and col_idx == 1:
                ax[row_idx, col_idx].set_title(loss_type)

    plt.show()

    # save plots on disk
    # fig.suptitle(loss_type)
    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(sys.path[0], '20220324-230759_51_lstm_0.783_0.071', 'loss_plot.jpg'))

    print('Created plots for the result files.')
