# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:34:49 2024

@author: zmzhai
"""
# just replace the transformer encoder with xlstm encoder in the main file

import numpy as np
import torch
import torch.nn as nn
import math

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class xLSTM(nn.Module):
    """
    A demonstration of using two multi-layer LSTMs (with dropout),
    plus a simple residual (skip) connection between them.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        # A learnable mean for each feature to fill missing data
        self.impute_mean = nn.Parameter(torch.zeros(input_size))
        
        # First LSTM: input_size -> hidden_size
        #   multi-layer with dropout between layers
        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Second LSTM: hidden_size -> hidden_size
        #   must match LSTM1's hidden_size output, or else shape mismatch
        self.lstm2 = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Optional final linear layer
        if output_size is not None:
            self.fc_out = nn.Linear(hidden_size, output_size)
        else:
            self.fc_out = None

    def forward(self, x):
        """
        Args:
            x:    [batch, seq_len, input_size]
            mask: [batch, seq_len, input_size] (1=observed, 0=missing)

        Returns:
            out:  [batch, seq_len, output_size] if output_size is set,
                  else [batch, seq_len, hidden_size]
        """
        # 1) Naive missing-data fill with a learnable mean
        # x_filled = mask * x + (1 - mask) * self.impute_mean.view(1, 1, -1)
        
        # 2) Pass through the first multi-layer LSTM
        #    - out1: [batch, seq_len, hidden_size]
        #    - (hn1, cn1): shape [num_layers, batch, hidden_size] each
        out1, (hn1, cn1) = self.lstm1(x)

        # 3) Pass through the second multi-layer LSTM
        #    - out2: [batch, seq_len, hidden_size]
        out2, (hn2, cn2) = self.lstm2(out1)

        # 4) Simple skip connection: out2 + out1
        #    - Both are [batch, seq_len, hidden_size]
        out2 = out2 + out1

        # 5) Optional final projection
        if self.fc_out is not None:
            out = self.fc_out(out2)  # e.g., [batch, seq_len, output_size]
        else:
            out = out2  # [batch, seq_len, hidden_size]

        return out




















