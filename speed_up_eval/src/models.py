import torch.nn as nn
import torch.nn.functional as F

class EMNISTModel(nn.Module):
    def __init__(self, num_classes=47):
        super(EMNISTModel, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # Fully connected layer: 7*7*64 inputs, 2048 outputs
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=2048)
        # Output layer: 2048 inputs, num_classes outputs
        self.fc2 = nn.Linear(in_features=2048, out_features=num_classes)
        # Max pooling layer with a 2x2 window
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout layer (optional, helps prevent overfitting)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply first convolution, followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: [batch_size, 32, 14, 14]
        # Apply second convolution, followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Output: [batch_size, 64, 7, 7]
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 7*7*64)
        # Apply first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        # Apply output layer
        logits = self.fc2(x)
        return logits
