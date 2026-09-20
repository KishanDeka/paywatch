import torch
from torch import nn
from torch.nn import functional as F


class LSTMCell(nn.Module):
    """Four gates implemented directly; autograd supplies differentiation."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        for block in self.weight_hh.chunk(4, dim=0):
            nn.init.orthogonal_(block)
        with torch.no_grad():
            self.bias[hidden_size : 2 * hidden_size].fill_(1.0)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (F.linear(x, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)).chunk(
            4, dim=-1
        )
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class LSTMAutoencoder(nn.Module):
    def __init__(self, features=30, hidden=24, length=10):
        super().__init__()
        if min(features, hidden, length) < 1:
            raise ValueError("dimensions must be positive")
        self.features, self.hidden, self.length = features, hidden, length
        self.encoder = LSTMCell(features, hidden)
        self.decoder = LSTMCell(hidden, hidden)
        self.output = nn.Linear(hidden, features)

    def forward(self, x):
        if not torch.jit.is_tracing() and (
            x.ndim != 3 or x.shape[1:] != (self.length, self.features)
        ):
            raise ValueError("expected [batch, sequence_length, features]")
        h = x.new_zeros((x.shape[0], self.hidden))
        c = torch.zeros_like(h)
        for t in range(self.length):
            h, c = self.encoder(x[:, t], (h, c))
        latent = h
        h, c = torch.zeros_like(h), torch.zeros_like(c)
        reconstructed = []
        for _ in range(self.length):
            h, c = self.decoder(latent, (h, c))
            reconstructed.append(self.output(h))
        return torch.stack(reconstructed, dim=1)
