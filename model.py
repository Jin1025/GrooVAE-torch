import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader
import numpy as np
from time import time

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)  # Bidirectional이므로 *2
        self.fc_std = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n은 (num_layers * num_directions, batch, hidden_size)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # Bidirectional의 마지막 layer들 concat
        mu = self.fc_mu(h_n)
        std = self.fc_std(h_n)
        z = self.reparameterize(mu, std)
        return z, mu, std

    def reparameterize(self, mu, std):
        std = torch.exp(0.5 * std)  
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z, seq_len):
        batch_size = z.size(0)
        h, c = self.init_hidden(batch_size)

        outputs = []
        input = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, latent_dim)
        
        out, (h, c) = self.lstm(input, (h, c))
        out = self.fc(out)
        output_hits, output_velocities, output_offsets = torch.chunk(out, 3, dim=-1)
        
        temperature = 1.0
        bernoulli_dist = dist.Bernoulli(logits=output_hits / temperature)
        sampled_hits = bernoulli_dist.sample()
        
        output_velocities = torch.sigmoid(output_velocities)
        output_offsets = torch.tanh(output_offsets)
        
        output = torch.cat([sampled_hits, output_velocities, output_offsets], dim=-1)
        
        return output, output_hits, output_velocities, output_offsets

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)
        return h, c

    def compute_loss(self, x_target, output_hits, output_velocities, output_offsets):
        target_hits, target_velocities, target_offsets = torch.chunk(x_target, 3, dim=-1)
        
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # pos_weight = torch.ones([32, 9], device=device) * 0.5
        hits_loss = F.binary_cross_entropy_with_logits(output_hits, target_hits, reduction='none').sum(dim=-1) # , pos_weight=pos_weight
        velocities_loss = F.mse_loss(output_velocities, target_velocities, reduction='none').sum(dim=-1)
        offsets_loss = F.mse_loss(output_offsets, target_offsets, reduction='none').sum(dim=-1)
        loss = hits_loss + velocities_loss + offsets_loss
        
        return loss.mean()

