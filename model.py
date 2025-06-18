import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    




class PositionalEncoding(nn.Module):
    # seq_len is number of token in a sentence
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # Percentage of Deactivated neurons

        # Create a matrix of shape (seq_len, d_model)
        positional_encoding = torch.zeros(seq_len, d_model)

        # Temp tensor to help Replicate the formula in the paper
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        even_index = torch.arange(0, d_model, 2)
        odd_index = torch.arange(1, d_model, 2)

        # assets/PE_LogTerm.png
        div_term = torch.exp(even_index * -math.log(10000)/d_model)

        # Formula from the paper section 3.5
        positional_encoding[:, even_index] += torch.sin(position * div_term)
        positional_encoding[:, odd_index] += torch.cos(position * div_term)

        # Add a batch dimension (1, seq_len, d_model)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding )

    def forward(self, x):
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps # Avoid 0 Division
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        # Originally (std + self.eps) was math.sqrt(std**2 + self.eps), but we use this because it's approximately the same anyway
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    # In Paper d_model = 512 , d_ff = 20048
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # tensor shape for each layer passthrough
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))