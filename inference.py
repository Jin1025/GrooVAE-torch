import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
import pickle
import random
import pretty_midi

from test import *
from model import *
from drum_utils import *

# Load data
path = '/workspace/GrooVAE/data/data_processed/'
with open(path + 'tapify_test.pkl', 'rb') as f:
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

# real_inference
data = test_data[100] # (32, 54)
input_data = data[:, :27]
print(input_data)

target_data = data[:, 27:] # (32, 27)
target_data = torch.tensor(target_data)
target_data = from_tensors_to_midi(target_data)
# target_data.write('/workspace/GrooVAE/data/inference/sample_tap_tar_1.mid')

with torch.no_grad():
    input_data = input_data.unsqueeze(0) # (1, 32, 27)
    input_data = input_data.to(device)
    seq_len = input_data.size(1)
    
    z, mu, std = model_encoder(input_data)
    output, output_hits, output_velocities, output_offsets = model_decoder(z, seq_len)

output = output.cpu().numpy()
midi_data = from_tensors_to_midi(output[0])
midi_data.write('/workspace/GrooVAE/data/inference/sample_tap_gen_1.mid')

# test_inference
# output_tensor = []

# with torch.no_grad():
#     for batch_idx, data in enumerate(test_set):
#         data = data.to(device)
#         seq_len = data.size(1)
            
#         x_test = data[:, :, :27] 
#         x_target = data[:, :, 27:]
            
#         z, mu, std = model_encoder(x_test)
#         output, output_hits, output_velocities, output_offsets = model_decoder(z, seq_len)
            
#         output_tensor.append(output.cpu())
        
# output_tensor = torch.cat(output_tensor, dim=0)

# midi_data = output_tensor[10].numpy() # (32,27)
# midi_data = from_tensors_to_midi(midi_data)
# midi_data.write('/workspace/GrooVAE/data/inference/sample_hum_gen_5.mid') # colab에서 fluidsynth로..

# input_data = x_test[10]
# print(input_data)

# target_data = x_target[10].cpu().numpy()
# target_data = from_tensors_to_midi(target_data)
# # target_data.write('/workspace/GrooVAE/data/inference/sample_target_2.mid')
# target_note = []

# for instrument in target_data.instruments:
#     instrument_info = {'program': instrument.program, 'is_drum': instrument.is_drum, 'notes': []}
#     for note in instrument.notes:
#         note_data = {
#             "pitch": note.pitch,
#             "start": note.start,
#             "end": note.end,
#             "velocity": note.velocity
#         }
#         instrument_info['notes'].append(note_data)
#     target_note.append(instrument_info)
    
# print(target_note)