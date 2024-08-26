import pickle
import random
import pretty_midi
import numpy as np
from time import time

import IPython.display
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchsummary import summary as summary

from drum_utils import *
from train import *
from test import *
from model import *

# initialize model with GPU
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# Load data and shuffle
path = '/workspace/GrooVAE/data/data_processed/'
# file_names = ['humanize_train.pkl', 'humanize_valid.pkl', 'humanize_test.pkl']
file_names = ['tapify_train.pkl', 'tapify_valid.pkl', 'tapify_test.pkl']
data_names = ['train_data', 'val_data', 'test_data']

for file_name, data_name in zip(file_names, data_names):
    with open(path + file_name, 'rb') as f:
        data = pickle.load(f)
        random.shuffle(data)
        globals()[data_name] = data
        print(f'The number of data in {data_name}: {len(data)}')

# dataloader
class DatasetSampler(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(torch.float32)

params = {'batch_size': 512, 
          'shuffle': True,
          'pin_memory': True,
          'num_workers': 4}

train_set = DataLoader(DatasetSampler(train_data), **params)
val_set = DataLoader(DatasetSampler(val_data), **params)
test_set = DataLoader(DatasetSampler(test_data), **params)

# init model
enc_input_size = 27
enc_hidden_size = 512
enc_latent_dim = 256

encoder = Encoder(enc_input_size, enc_hidden_size, enc_latent_dim)
encoder = encoder.to(device)

dec_input_size = 256
dec_hidden_size = 256
dec_output_size = 27

decoder = Decoder(dec_input_size, dec_hidden_size, dec_output_size)
decoder = decoder.to(device)

model = [encoder, decoder]

# init optimizer
enc_optimizer = optim.Adam(encoder.parameters(), lr=1e-3) # lr=1e-6
dec_optimizer = optim.Adam(decoder.parameters(), lr=1e-3) # lr=1e-6
optimizer = [enc_optimizer, dec_optimizer]

# Train
history_train = groove_train(device, train_set, val_set, model, optimizer, epochs=100)

# save model
# model_path = '/workspace/GrooVAE/model/humanize_2bar_16quant_5th_bcewithlogits'
model_path = '/workspace/GrooVAE/model/tapify_2bar_16quant_1st'
torch.save(encoder.state_dict(), model_path+'_encoder.pt')
torch.save(decoder.state_dict(), model_path+'_decoder.pt')

# Test
history_test = groove_test(device, test_set, model) # , temp = 1, options='full_sampling'

def plot_loss_history(history, start_epoch=4):
    # Define the starting epoch index
    start_idx = start_epoch - 1  # Convert epoch to index (0-based)
    
    # Ensure that the start index is within bounds
    if start_idx < 0 or start_idx >= len(history['train_loss']):
        raise ValueError("start_epoch is out of bounds of the history data.")
    
    # Slice the data to start from the specified epoch
    epochs = range(start_epoch, len(history['train_loss']) + 1)
    train_losses = history['train_loss'][start_idx:]
    val_losses = history['val_loss'][start_idx:]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Train')
    plt.plot(epochs, val_losses, 'r-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(False)  
    plt.savefig(model_path + '_loss.png')
    
plot_loss_history(history_train)