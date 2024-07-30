import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from dataloaders.Dataloader_Sensors import get_sensor_data, get_eyetracking_df, get_timestamps, merge_dfs, add_label

# Define Generator with LSTM layers
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, signal_length):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.LeakyReLU(0.2)
        )
        self.lstm = nn.LSTM(128, 16, batch_first=True, num_layers=2)
        self.fc_out = nn.Linear(16, signal_length)
        
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat((noise, label_embedding), dim=1)
        x = self.fc(x)
        x = x.unsqueeze(1).repeat(1, 15, 1)
        x, _ = self.lstm(x)
        generated_signal = self.fc_out(x[:, -1, :])
        return generated_signal

# Define Discriminator with temporal convolutional layers
class Discriminator(nn.Module):
    def __init__(self, signal_length, label_dim):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, signal, labels):
        label_embedding = self.label_embedding(labels)
        signal = signal.unsqueeze(1)
        validity = self.model(signal)
        return validity

# Hyperparameters
latent_dim = 100
signal_length = 15  # Assuming the time window is 15 seconds
label_dim = 2  # 0 for cybersickness, 1 for non-cybersickness
batch_size = 32
epochs = 1650
lr = 0.0002

# Load Dataset
class SensorData(Dataset):
    def __init__(self, filepath, use_eyetracking_data, use_sensor_data, buffer, buffer_cs_timestamps,
                 overwrite, timesteps, algorithm):
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
                        user_dfs.append(get_sensor_data(self.filepath, person, self.buffer))
                    if use_eyetracking_data:
                        user_dfs.append(get_eyetracking_df(self.filepath, person))
                    self.label_df = get_timestamps(filepath, person)
                    user_df_merged = merge_dfs(user_dfs, self.buffer)
                    user_df_merged = add_label(user_df_merged, self.label_df, self.buffer_cs_timestamps)
                    user_df_merged.dropna(inplace=True)
                    df_list_all_users.append(user_df_merged)
            self.df_all_users = pd.concat(df_list_all_users)
            pickle.dump(self.df_all_users, open(os.path.join(filepath, file_name), "wb"))
        else:
            with open(os.path.join(filepath, file_name), "rb") as data:
                self.df_all_users = pickle.load(data)

        x = self.df_all_users.drop(columns=['label']).values
        y = self.df_all_users['label'].values

        x = Variable(torch.tensor(x, dtype=torch.float32))
        y = Variable(torch.tensor(y, dtype=torch.float32))


        self.features = x
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

# Instantiate models
generator = Generator(latent_dim, label_dim, signal_length)
discriminator = Discriminator(signal_length, label_dim)

# Loss and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Create DataLoader
dataset = SensorData(filepath='../data/train', use_eyetracking_data=False, use_sensor_data=True, buffer=10, buffer_cs_timestamps=5, overwrite=False, timesteps=10, algorithm='cnn')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, (real_signals, labels) in enumerate(dataloader):
        batch_size = real_signals.size(0)
        
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)
        
        real_signals, labels, valid, fake = real_signals, labels.long(), valid, fake

        # Train Generator
        optimizer_G.zero_grad()
        
        noise = torch.randn(batch_size, latent_dim)
        gen_labels = torch.randint(0, 2, (batch_size,))
        gen_signals = generator(noise, gen_labels)
        
        g_loss = adversarial_loss(discriminator(gen_signals, gen_labels), valid)
        
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        
        real_loss = adversarial_loss(discriminator(real_signals, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_signals.detach(), gen_labels), fake)
        
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
    
    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item()}  G_loss: {g_loss.item()}")

print("Training finished.")
