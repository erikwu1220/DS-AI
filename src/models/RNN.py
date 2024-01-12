class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        # RNN layer
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True, nonlinearity='relu')

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input to have feature dimension of 1
        x = x.unsqueeze(-1)   # Assuming input x has shape (batch, sequence)

        # RNN layer
        x, hn = self.rnn(x)   # We do not need the hidden states hn

        # Select the output of the last time step
        x = x[:, -1, :]

        # Output layer
        x = self.fc(x)

        return x

import torch.nn as nn

class AdvancedRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_steps):
        super(AdvancedRNN, self).__init__()
        # RNN layer
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True, nonlinearity='relu')

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        self.num_steps = num_steps

    def forward(self, x):
        # Reshape input to have feature dimension of 1
        x = x.unsqueeze(-1)   # Assuming input x has shape (batch, sequence)

        # RNN layer
        x, hn = self.rnn(x)   # We do not need the hidden states hn

        # Select the output of the last time step
        x = x[:, -1, :]

        # Output layer for multi-step ahead prediction
        outputs = []
        for _ in range(self.num_steps):
            x = self.fc(x)
            outputs.append(x.unsqueeze(1))  # Add a dimension for the time step
            x, hn = self.rnn(x.unsqueeze(1), hn)  # Use the predicted value as the input for the next time step

        # Concatenate the predictions for all future time steps
        outputs = torch.cat(outputs, dim=1)

        return outputs