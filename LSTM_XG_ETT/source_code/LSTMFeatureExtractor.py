from torch import nn

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size = 64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]  