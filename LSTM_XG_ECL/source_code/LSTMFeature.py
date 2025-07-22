import torch 
import torch.nn as nn

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.output = nn.Linear(hidden_size, 256)  # tăng vector đặc trưng
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.output(h_n[-1])