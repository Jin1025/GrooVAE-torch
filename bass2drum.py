import pandas as pd
import pretty_midi
import pickle
from drum_utils import *

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
import random

from test import *
from model import *

file_name = '/workspace/GrooVAE/data/inference/bass_midi.mid'
save_path = '/workspace/GrooVAE/data/data_processed/'

# preprocess
bass_sample = []

midi_data = pretty_midi.PrettyMIDI(file_name)
inst = midi_data.instruments[0]
start_time = midi_data.get_onsets()[0]
beats = midi_data.get_beats(start_time)
        
if len(beats) < 2:
    raise ValueError("Insufficient number of beats")
        
fs = change_fs(beats, target_beats=16)
        
seqs_tensor, input_tensor, combined_tensor = to_tensors(inst, fs, start_time, tapify=True, fixed_velocities=True)

bass_sample.append(combined_tensor)

with open(save_path + 'bass_sample.pkl', 'wb') as f:
    pickle.dump(bass_sample, f)
    
print("bass2drum Done!")

with open(save_path + 'bass_sample.pkl', 'rb') as f:
        test_data = pickle.load(f)
        
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

test_set = DataLoader(DatasetSampler(test_data), **params)

# initialize model with GPU
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# init model
enc_input_size = 27
enc_hidden_size = 512
enc_latent_dim = 256

dec_input_size = 256
dec_hidden_size = 256
dec_output_size = 27

# load model
model_path = '/workspace/GrooVAE/model/tapify_2bar_16quant_1st'
model_encoder = Encoder(enc_input_size, enc_hidden_size, enc_latent_dim)
model_encoder = model_encoder.to(device)
model_encoder.load_state_dict(torch.load(model_path+'_encoder.pt'))
model_encoder.eval()

model_decoder = Decoder(dec_input_size, dec_hidden_size, dec_output_size)
model_decoder = model_decoder.to(device)
model_decoder.load_state_dict(torch.load(model_path+'_decoder.pt'))
model_decoder.eval()

# inference
data = test_data[0][3] # (32,54)
data = torch.tensor(data)
input_data = data[:, :27]
target_data = data[:, 27:] # (32,27)
target_data = from_tensors_to_midi(target_data)

with torch.no_grad():
    input_data = input_data.unsqueeze(0) # (1,32,27)
    input_data = input_data.to(device)
    seq_len = input_data.size(1)
    
    z, mu, std = model_encoder(input_data)
    output, output_hits, output_velocities, output_offsets = model_decoder(z, seq_len)

output = output.cpu().numpy()
midi_data = from_tensors_to_midi(output[0])
midi_data.write('/workspace/GrooVAE/data/inference/bass2drum_sample3.mid')