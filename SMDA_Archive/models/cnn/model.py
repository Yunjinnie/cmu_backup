import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_CNN(nn.Module):
    def __init__(self, n_out):
        super(CNN_CNN, self).__init__()

        # CNN block for visual data
        self.cnn_block_v = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 2048),
            nn.ReLU()
        )

        # CNN block for audio data (spectrogram)
        # Simplified CNN block for audio data (spectrogram)
        self.cnn_block_a = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),  # Adjusted to produce a 128-dimensional output
            nn.ReLU()
        )


        # Combined layers
        self.combined_fc = nn.Sequential(
            # nn.Linear(4096 + 1024, 2048),
            # nn.ReLU(),
            nn.Linear(2048+512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_out),
            nn.Sigmoid()
        )

    def forward(self, cnn_inp_v, cnn_inp_a):
        # Process visual input through the CNN block
        cnn_out_v = self.cnn_block_v(cnn_inp_v)

        # Process audio input through the CNN block
        cnn_out_a = self.cnn_block_a(cnn_inp_a)

        # Combine features from both modalities
        combined_inp = torch.cat((cnn_out_v, cnn_out_a), 1)
        out = self.combined_fc(combined_inp)

        return out
