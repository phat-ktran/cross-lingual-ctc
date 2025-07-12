import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    A simple CRNN model for CTC loss.

    The model consists of a CNN backbone for feature extraction, followed by
    a bidirectional RNN for sequence modeling, and a final linear layer
    for transcription.

    Args:
        in_channels (int): Number of channels in the input image (e.g., 1 for grayscale).
        num_classes (int): Number of output classes for the transcription, including the blank token.
        cnn_out_channels (int): The number of channels produced by the CNN.
        rnn_hidden_size (int): The number of features in the RNN hidden state.
        rnn_num_layers (int): The number of recurrent layers in the RNN.
    """

    def __init__(
        self,
        in_channels=3,
        vocab_size=6625,
        cnn_out_channels=256,
        rnn_hidden_size=256,
        rnn_num_layers=2,
        out_char_num=40,
    ):
        super(CRNN, self).__init__()

        self.cnn_backbone = nn.Sequential(
            # --- CNN Backbone ---
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves height and width
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves height and width again
            # Block 3
            nn.Conv2d(128, cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(
                cnn_out_channels, cnn_out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            # This adaptive pooling layer collapses the height dimension to 1.
            # This makes the model robust to variations in input image height.
            nn.AdaptiveAvgPool2d((1, out_char_num)),  # Output shape: (B, C, 1, W_new)
        )

        # --- RNN Sequence Modeling ---
        self.rnn = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            batch_first=False,  # Expects input of shape (seq_len, batch, input_size)
            dropout=0.5 if rnn_num_layers > 1 else 0,
        )

        # --- Transcription Layer ---
        self.fc = nn.Linear(
            rnn_hidden_size * 2,  # *2 because the RNN is bidirectional
            vocab_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CRNN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, sequence_length, num_classes).
                          These are ready to be used with a CTC loss function.
        """
        # 1. Feature extraction with CNN
        # Input shape: (batch_size, in_channels, height, width)
        features = self.cnn_backbone(x)
        # Output shape: (batch_size, cnn_out_channels, 1, sequence_length)

        # 2. Prepare features for RNN
        # Squeeze the height dimension and permute to fit RNN input format
        b, c, h, w = features.size()
        assert h == 1, f"The height of the CNN output feature map must be 1 (is: {h})"
        features = features.squeeze(2)  # Shape: (batch, channels, width)
        features = features.permute(
            2, 0, 1
        )  # Shape: (width, batch, channels) -> (seq_len, batch, input_size)

        # 3. Sequence modeling with RNN
        # Input shape: (seq_len, batch, input_size)
        rnn_output, _ = self.rnn(features)
        # Output shape: (seq_len, batch, rnn_hidden_size * 2)

        # 4. Transcription
        # Input shape: (seq_len, batch, rnn_hidden_size * 2)
        output = self.fc(rnn_output)
        # Output shape: (seq_len, batch, num_classes)

        # 5. Permute for CTC loss
        # The standard format for many CTC loss implementations is (batch, seq_len, num_classes)
        output = output.permute(1, 0, 2)

        return output
