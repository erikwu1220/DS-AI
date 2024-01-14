import torch.nn as nn

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

class CNNBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False, batch_norm=True):
            super().__init__()

            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.PReLU())
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias))

            self.cnnblock = nn.Sequential(*layers)

        def forward(self, x):
            return self.cnnblock(x)

class Encoder(nn.Module):
    def __init__(self, channels=[32, 64, 128], kernel_size=3, padding=1, bias=False, batch_norm=True):
        super().__init__()

        self.enc_blocks = nn.ModuleList([
            CNNBlock(channels[block], channels[block + 1], kernel_size, padding, bias,
                        batch_norm=batch_norm)
            for block in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        outs = []
        for block in self.enc_blocks:
            x = block(x)
            outs.append(x)
            x = self.pool(x)
        return outs


class Decoder(nn.Module):
    def __init__(self, channels=[128, 64, 32], kernel_size=3, padding=1, bias=False, batch_norm=True):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[block], channels[block + 1], kernel_size=2, padding=0, stride=2)
            for block in range(len(channels) - 1)]
        )
        self.dec_blocks = nn.ModuleList([
            CNNBlock(channels[block], channels[block + 1], kernel_size, padding, bias,
                     batch_norm=batch_norm)
            for block in range(len(channels) - 1)]
        )

    def forward(self, x, x_skips):
        for i in range(len(x_skips)):
            x = self.upconvs[i](x)
            x = torch.cat((x, x_skips[-(1 + i)]), dim=1)
            x = self.dec_blocks[i](x)

        x = self.dec_blocks[-1](x)
        return x


class CNN(nn.Module):
    def __init__(self, node_features, out_dim=1, n_downsamples=3, initial_hid_dim=64, batch_norm=True,
                 bias=True):
        super(CNN, self).__init__()
        hidden_channels = [initial_hid_dim*2**i for i in range(n_downsamples)]
        encoder_channels = [node_features]+hidden_channels
        decoder_channels = list(reversed(hidden_channels))+[out_dim]

        self.encoder = Encoder(encoder_channels, kernel_size=3, padding=1,
                               bias=bias, batch_norm=batch_norm)
        self.decoder = Decoder(decoder_channels, kernel_size=3, padding=1,
                               bias=bias, batch_norm=batch_norm)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x[-1], x[:-1])
        x = nn.Sigmoid()(x)
        return x


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

