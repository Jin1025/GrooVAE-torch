import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from time import time

def groove_test(device, test_loader, model):
    history = {'test_loss': []}
    
    encoder, decoder = model
    encoder.eval()
    decoder.eval()
    
    test_loss = 0
    start_time = time() 
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            batch_size = data.size(0)
            seq_len = data.size(1)
            
            x_test = data[:, :, :27]  
            x_test_target = data[:, :, 27:]  
            
            # Forward pass through encoder
            z, x_test_mu, x_test_std = encoder(x_test)
            
            # Forward pass through decoder
            output, output_hits, output_velocities, output_offsets = decoder(z, seq_len)
            
            # Loss calculation
            reconstruction_loss = decoder.compute_loss(x_test_target, output_hits, output_velocities, output_offsets)
            logvar = x_test_std.pow(2).log()
            kl_loss = -0.5 * torch.sum(1 + logvar - x_test_mu.pow(2) - logvar.exp())
            beta = 0.2
            loss = reconstruction_loss + beta * kl_loss
            
            test_loss += loss.item()
    
    test_loss = test_loss / (batch_idx + 1)
    
    history['test_loss'].append(test_loss)
    
    print(f'({time() - start_time:.2f} sec) - test_loss: {test_loss:.3f}')
    
    return history
