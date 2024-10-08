# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:25:18 2024

@author: zmzhai
"""

import torch
import torch.nn as nn
import math

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        # Embedding layer for positional encoding
        self.pos_encoder = PositionalEncoding(d_model, 50000)
        # Transformer layer
        transformer_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                        dim_feedforward=dim_feedforward, 
                                                        dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, num_layers=num_layers)
        # Linear layer to map the output
        self.decoder = nn.Linear(d_model, output_size)
        self.init_weights()
        self.d_model = d_model
        
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, return_attention=False, return_qkv=False):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        
        # Generate a square subsequent mask for causal attention
        # device = src.device
        # causal_mask = self.generate_square_subsequent_mask(src.size(1)).to(device)
        # output = self.transformer_encoder(src, mask=causal_mask)
        
        # without casual attention mask
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int) -> None:
        super().__init__()

        p_pos = torch.arange(0, seq_length).unsqueeze(1).to(device)
        p_i = torch.arange(0, d_model).to(device)

        PE = (p_pos / (1000**(2*p_i/d_model))).unsqueeze(0)
        PE[0, :, 0::2] = torch.sin(PE[:, :, 0::2])
        PE[0, :, 1::2] = torch.cos(PE[:, :, 1::2])
        self.PE = PE.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.PE[:, :x.shape[1], :]


class SmoothnessLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(SmoothnessLoss, self).__init__()
        self.alpha = alpha  # Weight for Laplacian regularization
        self.beta = beta    # Weight for Total Variation regularization

    def forward(self, predictions):
        # Compute second-order differences (Laplacian Regularization)
        laplacian = predictions[:, :-2] - 2 * predictions[:, 1:-1] + predictions[:, 2:]
        laplacian_loss = torch.mean(laplacian ** 2)

        # Compute absolute differences (Total Variation Regularization)
        total_variation = torch.abs(predictions[:, 1:] - predictions[:, :-1])
        total_variation_loss = torch.mean(total_variation)

        # Combine the losses
        return self.alpha * laplacian_loss + self.beta * total_variation_loss
    

# Define the combined loss function
def combined_loss_function(outputs, targets, smoothness_loss_fn, mse_weight=1.0, smoothness_weight=0.1):
    # Calculate MSE loss
    mse_loss = nn.MSELoss()(outputs, targets)
    
    # Calculate smoothness loss
    smoothness_loss = smoothness_loss_fn(outputs)
    
    # Combine the losses
    total_loss = mse_weight * mse_loss + smoothness_weight * smoothness_loss
    return total_loss















