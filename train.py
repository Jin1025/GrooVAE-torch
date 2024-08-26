import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from time import time

def groove_train(device, train_loader, val_loader, model, optimizer, epochs=100):
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    encoder, decoder = model
    enc_optimizer, dec_optimizer = optimizer
    
    enc_scheduler = optim.lr_scheduler.CosineAnnealingLR(enc_optimizer, epochs, eta_min=1e-6)
    dec_scheduler = optim.lr_scheduler.CosineAnnealingLR(dec_optimizer, epochs, eta_min=1e-6)

    for epoch in range(1, epochs + 1):
        start_time = time()
        
        train_loss = 0
        val_loss = 0
        
        # Train
        encoder.train()
        decoder.train()
        
        for batch_idx, data in enumerate(train_loader):
            
            data = data.to(device)
            batch_size = data.size(0)
            seq_len = data.size(1)
            
            x_train = data[:, :, :27]   
            x_train_target = data[:, :, 27:]  
            
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            
            # Encoder
            z, x_train_mu, x_train_std = encoder(x_train)
            
            # Decoder
            output, output_hits, output_velocities, output_offsets = decoder(z, seq_len)
            
            # Loss calculation
            reconstruction_loss = decoder.compute_loss(x_train_target, output_hits, output_velocities, output_offsets)
            logvar = x_train_std.pow(2).log()
            kl_loss = -0.5 * torch.sum(1 + logvar - x_train_mu.pow(2) - logvar.exp())
            beta = 0.2
            loss = reconstruction_loss + beta * kl_loss 
            
            # Backward
            loss.backward()
            enc_optimizer.step() 
            dec_optimizer.step()
            
            train_loss += loss.item()
        
        enc_scheduler.step()
        dec_scheduler.step()
        
        train_loss = train_loss / (batch_idx + 1)
        history['train_loss'].append(train_loss)

        # Validation
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device)
                batch_size = data.size(0)
                seq_len = data.size(1)
                
                x_val = data[:, :, :27]  
                x_val_target = data[:, :, 27:]  
                
                # Forward pass
                z, x_val_mu, x_val_std = encoder(x_val)
                output, output_hits, output_velocities, output_offsets = decoder(z, seq_len)
                
                # Loss calculation
                reconstruction_loss = decoder.compute_loss(x_val_target, output_hits, output_velocities, output_offsets)
                logvar = x_val_std.pow(2).log()
                kl_loss = -0.5 * torch.sum(1 + logvar - x_val_mu.pow(2) - logvar.exp())
                beta = 0.2
                loss = reconstruction_loss + beta * kl_loss
                
                val_loss += loss.item()
        
        val_loss = val_loss / (batch_idx + 1)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch} ({time() - start_time:.2f} sec) - train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, lr: {enc_scheduler.get_last_lr()[0]:.6f}')

    return history
