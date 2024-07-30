

class ModelConfigurations:
    """Instead of input params in the script 'train_model.py': define the different configurations
    for the grid search here."""

    def __init__(self):
        self.configurations = []
        timesteps = [30, 50]
        buffers = ['0.1 S', '0.5 S']
        buffers_cs_timestamps = [500000, 1000000]
        hidden_sizes1 = [32, 64]
        hidden_sizes2 = [8, 16]
        learning_rates = [0.001, 0.005]
        # dropouts_lstm = [0.2, 0.5, 0.7]
        dropouts_p = [0.5, 0.7]
        algorithms = ['svm', 'lstm', 'cnn']

        # counter = 1
        # for algorithm in algorithms:
        #     if algorithm == 'svm':
        #         # there are limited options for svm
        #         for timestep in timesteps:
        #             for buffer in buffers:
        #                 for buffer_cs_timestamps in buffers_cs_timestamps:
        #                     self.add_configuration(counter, algorithm, True, True, None, None, None, None, None, 256,
        #                                            timestep, buffer, buffer_cs_timestamps)
        #                     counter += 1
        #     else:
        #         for timestep in timesteps:
        #             for buffer in buffers:
        #                 for buffer_cs_timestamps in buffers_cs_timestamps:
        #                     for hidden_size1 in hidden_sizes1:
        #                         for hidden_size2 in hidden_sizes2:
        #                             for learning_rate in learning_rates:
        #                                  for dropout_p in dropouts_p:
        #                                     self.add_configuration(counter, algorithm, True, True, 0.5, dropout_p,
        #                                                            learning_rate, hidden_size1, hidden_size2, 256,
        #                                                            timestep, buffer, buffer_cs_timestamps)
        #                                     counter += 1

        # best svm configuration
        self.add_configuration(4, 'svm', True, True, None, None, None, None, None, 256, 30, '0.5 S', 1000000)
        # best lstm configuration
        self.add_configuration(67, 'lstm', True, True, 0.5, 0.5, 0.005, 64, 8, 256, 30, '0.5 S', 1000000)
        # best cnn1 configuration
        # self.add_configuration(154, 'cnn', True, True, 0.5, 0.7, 0.001, 32, 8, 256, 30, '0.1 S', 1000000)
        # best cnn2 configuration
        self.add_configuration(230, 'cnn', True, True, 0.5, 0.7, 0.001, 64, 16, 256, 50, '0.1 S', 1000000)
        # best cnn3 configuration
        # self.add_configuration(192, 'cnn', True, True, 0.5, 0.7, 0.005, 32, 16, 256, 30, '0.5 S', 1000000)

    def add_configuration(self, id, algorithm, use_eyetracking, use_sensor_data,
                 dropout_lstm, dropout_p, learning_rate, hidden_size1, hidden_size2, batch_size, timesteps,
                          buffer, buffer_cs_timestamps):

        new_config = Configuration(id, algorithm, use_eyetracking, use_sensor_data,
                 dropout_lstm, dropout_p, learning_rate, hidden_size1, hidden_size2, batch_size, timesteps,
                                   buffer, buffer_cs_timestamps)

        self.configurations.append(new_config)


class Configuration:
    """Class for one configuration object."""

    def __init__(self, id, algorithm, use_eyetracking, use_sensor_data,
                 dropout_lstm, dropout_p, learning_rate, hidden_size1, hidden_size2, batch_size, timesteps,
                 buffer, buffer_cs_timestamps):
        self.id = id
        self.algorithm = algorithm  # svm, lstm or cnn
        self.use_eyetracking_data = use_eyetracking
        self.use_sensor_data = use_sensor_data
        self.dropout_lstm = dropout_lstm
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.buffer = buffer
        self.buffer_cs_timestamps = buffer_cs_timestamps

    def __str__(self):
        return (f'Configuration:\n'
            f'\tid: {self.id}\n'
            f'\talgorithm: {self.algorithm}\n'
            f'\tuse_eyetracking_data: {self.use_eyetracking_data}\n'
            f'\tuse_sensor_data: {self.use_sensor_data}\n'
            f'\tdropout_lstm: {self.dropout_lstm}\n'
            f'\tdropout_p: {self.dropout_p}\n'
            f'\tlearning_rate: {self.learning_rate}\n'
            f'\thidden_size1: {self.hidden_size1}\n'
            f'\thidden_size2: {self.hidden_size2}\n'
            f'\tbatch_size: {self.batch_size}\n'
            f'\ttimesteps: {self.timesteps}\n'
            f'\tbuffer: {self.buffer}\n'
            f'\tbuffer_cs_timestamps: {self.buffer_cs_timestamps}\n'
        )
