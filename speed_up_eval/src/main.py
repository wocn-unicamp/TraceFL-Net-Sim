import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from time import time

from models import EMNISTModel

def train_emnist(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(data)
        loss = criterion(logits, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * data.size(0)

        # Calculate predictions and accuracy
        _, predictions = torch.max(logits, 1)
        correct += (predictions == target).sum().item()
        total += data.size(0)

        if batch_idx % 100 == 99:  # Print every 100 batches
            print(f'Epoch {epoch} [{batch_idx +1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}')

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Train Epoch: {epoch} \tLoss: {epoch_loss:.4f} \tAccuracy: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

def main() -> None:
    # Argument parsing
    parser = argparse.ArgumentParser(description="PyTorch Training for speed-up evaluation in EMNIST and Shakespeare")
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='emnist',
                        choices=['emnist', 'shakespeare'],
                        help='EMNIST split to use (default: balanced)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    match args.dataset:
        case "emnist":
            # Define transformations for the training and testing data
            transform = transforms.Compose([
                transforms.Resize((28, 28)),  # Ensure images are 28x28
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize with EMNIST's mean and std
            ])

             # Load EMNIST dataset
            train_dataset = datasets.EMNIST(
                root='data',
                split='digits', # Options: 'digits' (10 classes) , 'balanced' (47 classes), 'letters' (26 classes)
                train=True,
                download=True,
                transform=transform
            )

            # Define DataLoaders
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=64,
                shuffle=True
            )

            num_classes = 10

            model = EMNISTModel(num_classes=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            for epoch in range(1, args.epochs + 1):
                start = time()
                train_emnist(model, device, train_loader, optimizer, criterion, epoch)
                print(f"Epoch {epoch} - Train Time: {time() - start} s")
                scheduler.step()
        case _:
            raise SystemError

if __name__ == "__main__":
    main()
