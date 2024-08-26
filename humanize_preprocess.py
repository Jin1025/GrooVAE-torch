import pandas as pd
import pretty_midi
import pickle
from drum_utils import *

path = '/workspace/GrooVAE/data/midi_data/groove/'
save_path = '/workspace/GrooVAE/data/data_processed/'

df = pd.read_csv('/workspace/GrooVAE/data/midi_data/groove/info.csv')

humanize_train = []
humanize_valid = []
humanize_test = []

# humanize version
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
        
        seqs_tensor, input_tensor, combined_tensor = to_tensors(inst, fs, start_time, humanize=True)
        
        if combined_tensor is not None:  
            if split == 'train':
                humanize_train.append(combined_tensor)
            elif split == 'validation':
                humanize_valid.append(combined_tensor)
            else:
                humanize_test.append(combined_tensor)
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        continue

# Concatenate the tensors along the first dimension (number of windows)
if humanize_train:
    humanize_train = torch.cat(humanize_train, dim=0)
if humanize_valid:
    humanize_valid = torch.cat(humanize_valid, dim=0)
if humanize_test:
    humanize_test = torch.cat(humanize_test, dim=0)

# save pickle
with open(save_path + 'humanize_train.pkl', 'wb') as f:
    pickle.dump(humanize_train, f)

with open(save_path + 'humanize_valid.pkl', 'wb') as f:
    pickle.dump(humanize_valid, f)

with open(save_path + 'humanize_test.pkl', 'wb') as f:
    pickle.dump(humanize_test, f)

print("Humanize Done!")



