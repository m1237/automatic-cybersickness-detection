import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sn


def calc_confusion_matrix(predictions, labels, path, eval_type):
    """ Calculates and plots the confusion matrix.

            Parameters:
                predictions (Tensor): The model's predictions
                labels (Tensor): The true labels
                path (string): The path where to save the plot
                eval_type (string): 'train' or 'validation'

            Returns:
                None

        """
    classes = ('Not Cybersick', 'Cybersick')

    # Build confusion matrix
    cf_matrix = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(path, f'confusion_matrix_{eval_type}.png'))


def calc_precision_recall(predictions, labels, path, model_type, eval_type):
    """ Calculates precision, recall, f1 score and saves the values in a file.

            Parameters:
                predictions (Tensor): The model's predictions
                labels (Tensor): The true labels
                path (string): The path where to save the data
                model_type (string): Which model (algorithm) is used
                eval_type (string): 'train' or 'validation'

            Returns:
                F1 value

        """
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average='binary')

    # store in file
    if os.path.isfile(os.path.join(path, f'{model_type}_metrics.csv')):
        # file already exists, append metrics
        with open(os.path.join(path, f'{model_type}_metrics.csv'), 'a') as metrics_file:
            metrics_file.write(f'{eval_type};{precision};{recall};{f1};{support}\n')
            metrics_file.close()
    else:
        # create file and also write header
        with open(os.path.join(path, f'{model_type}_metrics.csv'), 'w') as metrics_file:
            metrics_file.write(f'Type;Precision;Recall;F1;Support\n')
            metrics_file.write(f'{eval_type};{precision};{recall};{f1};{support}\n')
            metrics_file.close()
    return f1


def print_predictions(predictions, labels, path, model_type, eval_type):
    """ Prints the predictions to a file.

            Parameters:
                predictions (Tensor): The model's predictions
                labels (Tensor): The true labels
                path (string): The path where to save the data
                model_type (string): Which model (algorithm) is used
                eval_type (string): 'train' or 'validation'

            Returns:
                None

        """
    # store in file
    with open(os.path.join(path, f'{model_type}_{eval_type}_predictions.txt'), 'w') as predictions_file:
        predictions_file.write(f'{eval_type}_labels:\t\t')
        np.savetxt(predictions_file, labels.numpy(), newline=" ", fmt='%d')
        predictions_file.write('\n')
        predictions_file.write(f'{eval_type}_predictions:\t')
        np.savetxt(predictions_file, predictions.numpy(), newline=" ", fmt='%d')
        predictions_file.write('\n')
        predictions_file.close()


def evaluate_svm(model, model_type, data, path, eval_type):
    """ Starts the evaluation process for the SVM classifier.
         Parameters:
            model (nn): The trained model.
            path (string): The path where to save the data
            model_type (string): Which model (algorithm) is used
            data (DataLoader): The Dataloader containing the validation data
            eval_type (string): 'train' or 'validation'

        Returns:
            F1 value
    """
    print(f'Starting evaluation for {eval_type}...')

    # make predictions
    y_pred = torch.Tensor(model.predict(data.features))

    # print the predictions
    print_predictions(y_pred, data.labels, path, model_type, eval_type)

    # confusion matrix
    calc_confusion_matrix(y_pred, data.labels, path, eval_type)

    # precision, recall, f1 score
    f1 = calc_precision_recall(y_pred, data.labels, path, model_type, eval_type)

    print(f'Evaluation for {eval_type} finished.')
    return f1


def evaluate_nn(model, model_type, data, path, eval_type):
    """ Starts the evaluation process for the NN models (LSTM/CNN).
         Parameters:
            model (nn): The trained model.
            path (string): The path where to save the data
            model_type (string): Which model (algorithm) is used
            data (DataLoader): The Dataloader containing the validation data
            eval_type (string): 'train' or 'validation'

        Returns:
            None
    """
    print(f'Starting evaluation for {eval_type}...')

    # make predictions
    model.eval()

    with torch.no_grad():
        outputs = model.forward(data.dataset.features)
        y_pred = model.sigmoid(outputs)
        y_pred = (y_pred > 0.5).type(torch.uint8)

        # print the predictions
        print_predictions(y_pred.squeeze(), data.dataset.labels.squeeze(), path, model_type, eval_type)

        # confusion matrix
        calc_confusion_matrix(y_pred.squeeze(), data.dataset.labels.squeeze(), path, eval_type)

        # precision, recall, f1 score
        f1 = calc_precision_recall(y_pred.squeeze(), data.dataset.labels.squeeze(), path, model_type, eval_type)

        print(f'Evaluation for {eval_type} finished.')
        return f1


if __name__ == '__main__':
    """ Main method.
        Evaluates the model.

    Parameters:
        None
    Returns:
        None
    """

    # TODO: load model from disk, then do evaluation
    # evaluate(model, model_type, val_data, path)





