import argparse
import sys
import os

from tqdm import tqdm
from datetime import datetime
import torch.nn
from torch.utils.data import DataLoader
from sklearn import svm
import pickle

from classification.CS_LSTM import CSClassifierLSTM
from classification.CS_CNN_LSTM import CSClassifierCNNLSTM
from dataloaders.Dataloader_Sensors import SensorData
import evaluate_model as evaluation
from model_config import ModelConfigurations


def store_model_and_files(model_path, model_type, train_losses, val_losses):
    """ Stores the model configuration and additional files (train/val losses) on disk.

        Parameters:
            model_path (DataLoader): The data for training
            model_type (string): Which model (algorithm) is used
            train_losses ([Float]): The train loss array
            val_losses ([Float]): The validation loss array

        Returns:
            None

    """

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if model_type != 'svm':
        # store model
        torch.save({'model_state_dict': model.state_dict()},
                os.path.join(model_path, f'{model_type}.pth.tar'))

        # store model config json
        with open(os.path.join(model_path, f'{model_type}_layers.txt'), 'w') as config_file:
            config_file.write(model.__str__())
            config_file.close()
    else:
        # store model
        pickle.dump(model, open(os.path.join(model_path, f'{model_type}.pkl'), 'wb'))

    # store train/validation loss
    with open(os.path.join(model_path, f'{model_type}_losses.csv'), 'w') as loss_file:
        loss_file.write(f'Train Loss;Validation Loss\n')
        for index in range(len(train_losses)):
            loss_file.write(f'{train_losses[index]};{val_losses[index]}\n')
        loss_file.close()

    # store model config
    with open(os.path.join(model_path, f'{model_type}_config.csv'), 'w') as config_file:
        config_file.write(config.__str__())
        config_file.close()


def start_training_nn(train_data, val_data, n_iterations, model_type):
    """ Starts the training process.
            In each iteration, validation is done on the val_data.
            After training, the model is saved to disk.

        Parameters:
            train_data (DataLoader): The data for training
            val_data (DataLoader): The data for validation
            n_iterations (int): The number of iteration
            model_type (string): Which model is used

        Returns:
            None

        """
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(n_iterations)):
        model.train()
        epoch_train_loss = []

        for batch_idx, batch_sample in enumerate(train_data):
            features = batch_sample[0]
            labels = batch_sample[1]

            optimizer.zero_grad()  # Zero out any previously calculated gradients
            outputs = model.forward(features)  # forward pass

            # obtain the loss function
            loss = criterion(outputs, labels)

            loss.backward()  # calculates the loss of the loss function
            optimizer.step()  # improve from loss, i.e backprop

            epoch_train_loss.append(loss.item())

        if epoch % 5 == 0:
            train_loss_avg = sum(epoch_train_loss) / len(epoch_train_loss)
            print("Epoch: %d, loss: %1.5f" % (epoch, train_loss_avg))

            # store train loss
            train_losses.append(train_loss_avg)

            # do evaluation
            model.eval()

            # validation loss
            with torch.no_grad():
                outputs_val = model.forward(val_data.dataset.features)
                loss_val = criterion(outputs_val, val_data.dataset.labels)
                val_losses.append(loss_val.item())
                # y_pred = model.sigmoid(outputs_val)
                print("Epoch: %d, validation loss: %1.5f" % (epoch, loss_val.item()))

    # store models and additional files
    print(f"Save model to disk...")
    model_path = os.path.join(sys.path[0], 'models', f'{timestamp}_{model_id}_{model_type}')
    store_model_and_files(model_path, model_type, train_losses, val_losses)

    print(f'Training finished.')

    # evaluation for train and validation data
    f1_train = evaluation.evaluate_nn(model, model_type, train_data, model_path, 'train')
    f1_val = evaluation.evaluate_nn(model, model_type, val_data, model_path, 'validation')

    # append f1 to folder name so we can directly see which model is good
    os.rename(model_path, f'{model_path}_{f1_train:.3f}_{f1_val:.3f}')


def start_training_svm(train_data, val_data, model_type):
    """ Starts the training process.
            In each iteration, validation is done on the val_data.
            After training, the model is saved to disk.

        Parameters:
            train_data (Dataset): The data for training
            val_data (Dataset): The data for validation
            model_type (string): Which model is used

        Returns:
            None

        """
    train_losses = []
    val_losses = []

    features_train = train_data.features
    labels_train = train_data.labels

    # fit the model
    model.fit(features_train, labels_train)

    # make a prediction
    outputs = model.predict(features_train)

    # obtain the loss function
    loss = criterion(torch.Tensor(outputs), labels_train)

    print("Epoch: %d, loss: %1.5f" % (1, loss.item()))

    # store train loss
    train_losses.append(loss.item())

    # do evaluation
    features_val = val_data.features
    labels_val = val_data.labels

    # validation loss
    outputs_val = model.predict(features_val)
    loss_val = criterion(torch.Tensor(outputs_val), labels_val)
    val_losses.append(loss_val.item())
    print("Epoch: %d, validation loss: %1.5f" % (1, loss_val.item()))

    # store models and additional files
    print(f"Save model to disk...")
    model_path = os.path.join(sys.path[0], 'models', f'{timestamp}_{model_id}_{model_type}')
    store_model_and_files(model_path, model_type, train_losses, val_losses)

    print(f'Training finished.')

    # evaluation for train and validation data
    f1_train = evaluation.evaluate_svm(model, model_type, train_data, model_path, 'train')
    f1_val = evaluation.evaluate_svm(model, model_type, val_data, model_path, 'validation')

    # append f1 to folder name so we can directly see which model is good
    os.rename(model_path, f'{model_path}_{f1_train:.3f}_{f1_val:.3f}')


if __name__ == '__main__':
    """ Main method.
        Trains a model to predict cybersickness.

    Parameters:
            traindata_path (str): path were the training data is located (default: sys.path[0]/Storage/training)
            valdata_path (str): path were the validation data is located (default: sys.path[0]/Storage/validation)
            buffer (int): buffer for merging measurement data and cs timestamps
            overwrite (bool): if already created pickle files for the measurement data should be overwritten
            iterations (int): number of iterations for the training process
    Returns:
        None

    """

    # get input parameters
    print("Starting training script...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindata_path", help="path were the training data is located",
                        default=os.path.join(sys.path[0], 'Storage', 'training'))
    parser.add_argument("--valdata_path", help="path were the validation data is located",
                        default=os.path.join(sys.path[0], 'Storage', 'validation'))
    parser.add_argument("--overwrite", help="if already created pickle files for the measurement data should be overwritten",
                        default=True)
    parser.add_argument("--iterations", help="number of iterations for the training process",
                        default=30)

    args = parser.parse_known_args()[0]

    # get model configurations
    model_configs = ModelConfigurations()

    for config in model_configs.configurations:

        print(f'------------START TRAINING FOR CONFIGURATION {config.id}------------')
        model_id = config.id

        # load training data
        print("Loading training data...")
        datasets_train = SensorData(args.traindata_path, config.use_eyetracking_data, config.use_sensor_data,
                                    buffer=config.buffer, overwrite=args.overwrite, timesteps=config.timesteps,
                                    buffer_cs_timestamps=config.buffer_cs_timestamps, algorithm=config.algorithm)
        # load va data
        print("Loading validation data...")
        datasets_val = SensorData(args.valdata_path, config.use_eyetracking_data, config.use_sensor_data,
                                  buffer=config.buffer, overwrite=args.overwrite, timesteps=config.timesteps,
                                  buffer_cs_timestamps=config.buffer_cs_timestamps, algorithm=config.algorithm)

        # create dataloaders combining the different measurements
        dataloader_train = DataLoader(datasets_train, batch_size=config.batch_size, num_workers=0, shuffle=False)
        dataloader_val = DataLoader(datasets_val, batch_size=config.batch_size, num_workers=0, shuffle=False)

        # init training values
        num_classes = 1  # number of output classes (here: binary (cybersick = 1/not cybersick = 0))

        # timestamp to differentiate the trained models
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # init model
        print("Init model...")
        if config.algorithm == "svm":
            # init svm model
            model = svm.SVC(kernel="linear", class_weight={1: 8})
            # model = svm.SVC(kernel="rbf")
            # init loss function
            criterion = torch.nn.BCELoss()

            # start training process
            print("Starting training process...")
            start_training_svm(datasets_train, datasets_val, model_type=config.algorithm)
        elif config.algorithm == "lstm" or config.algorithm == "cnn":
            if config.algorithm == "lstm":
                # init LSTM model
                input_size = datasets_train.features.shape[2]  # number of features
                model = CSClassifierLSTM(num_classes, input_size, config.hidden_size1, config.hidden_size2,
                                            dropout_lstm=config.dropout_lstm, dropout_p=config.dropout_p, bidirectional = True)
            elif config.algorithm == "cnn":
                # input_size = 12 * config.hidden_size1  # for LSTM after CNN, 768 according to paper
                input_size = config.hidden_size1
                feature_size = datasets_train.features.shape[2]  # number of features
                model = CSClassifierCNNLSTM(num_classes, feature_size, input_size, config.hidden_size1, config.hidden_size2,
                                            dropout_lstm=config.dropout_lstm, dropout_p=config.dropout_p)
            # init loss function and optimizer
            criterion = torch.nn.BCEWithLogitsLoss()  # binary cross entropy loss with included sigmoid activation
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

            # start training process
            print("Starting training process...")
            start_training_nn(dataloader_train, dataloader_val,
                       n_iterations=args.iterations, model_type=config.algorithm)

        else:
            print(f'Skipping configuration {config.id}: value for "algorithm" has to be "svm", "lstm" or "cnn".')

    sys.exit("Training process finished.")


