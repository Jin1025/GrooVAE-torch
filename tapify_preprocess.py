import pandas as pd
import pretty_midi
import pickle
from drum_utils import *

path = '/workspace/GrooVAE/data/midi_data/groove/'
save_path = '/workspace/GrooVAE/data/data_processed/'

df = pd.read_csv('/workspace/GrooVAE/data/midi_data/groove/info.csv')

tapify_train = []
tapify_valid = []
tapify_test = []

# tapify version
for _, row in df.iterrows():
    file_name = path + row['midi_filename']
    split = row['split']
    
    try:
        midi_data = pretty_midi.PrettyMIDI(file_name)
        inst = midi_data.instruments[0]
        start_time = midi_data.get_onsets()[0]
        beats = midi_data.get_beats(start_time)
        
        if len(beats) < 2:
            raise ValueError("Insufficient number of beats")
        
        fs = change_fs(beats, target_beats=16)
        
        seqs_tensor, input_tensor, combined_tensor = to_tensors(inst, fs, start_time, tapify=True, fixed_velocities=True)
        
        if combined_tensor is not None:  
            if split == 'train':
                tapify_train.append(combined_tensor)
            elif split == 'validation':
                tapify_valid.append(combined_tensor)
            else:
                tapify_test.append(combined_tensor)
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        continue

# Concatenate the tensors along the first dimension (number of windows)
if tapify_train:
    tapify_train = torch.cat(tapify_train, dim=0)
if tapify_valid:
    tapify_valid = torch.cat(tapify_valid, dim=0)
if tapify_test:
    tapify_test = torch.cat(tapify_test, dim=0)

# save pickle
with open(save_path + 'tapify_train_2.pkl', 'wb') as f:
    pickle.dump(tapify_train, f)

with open(save_path + 'tapify_valid.pkl', 'wb') as f:
    pickle.dump(tapify_valid, f)

with open(save_path + 'tapify_test.pkl', 'wb') as f:
    pickle.dump(tapify_test, f)

print("Tapify Done!")
