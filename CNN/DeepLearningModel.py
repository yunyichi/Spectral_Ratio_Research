import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import helper_function
import torchvision.transforms as transforms
import numpy as np
import sys
import argparse

# Custom loss function to compute angular error in degrees
class AngularLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(AngularLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Predicted angles
        az_pred = y_pred[:, 0]
        el_pred = y_pred[:, 1]
        x_pred = torch.sin(el_pred) * torch.cos(az_pred)
        y_pred_vec = torch.sin(el_pred) * torch.sin(az_pred)
        z_pred = torch.cos(el_pred)
        pred_vector = torch.stack([x_pred, y_pred_vec, z_pred], dim=1)
        pred_vector = F.normalize(pred_vector, p=2, dim=1)

        # Ground truth angles
        az_true = y_true[:, 0]
        el_true = y_true[:, 1]
        x_true = torch.sin(el_true) * torch.cos(az_true)
        y_true_vec = torch.sin(el_true) * torch.sin(az_true)
        z_true = torch.cos(el_true)
        true_vector = torch.stack([x_true, y_true_vec, z_true], dim=1)
        true_vector = F.normalize(true_vector, p=2, dim=1)

        # Compute dot product
        dot_product = torch.sum(pred_vector * true_vector, dim=1)
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute angle in degrees
        angle_rad = torch.acos(dot_product)
        angle_deg = angle_rad * (180.0 / np.pi)

        if self.reduction == 'mean':
            return torch.mean(angle_deg)
        elif self.reduction == 'sum':
            return torch.sum(angle_deg)
        else:
            return angle_deg


# Custom transformations
class RandomCropTensor(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        _, h, w = image.shape
        th, tw = self.output_size
        if h < th or w < tw:
            raise ValueError("Input image is smaller than the output size.")
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return image[:, i:i + th, j:j + tw]

class CenterCropTensor(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        _, h, w = image.shape
        th, tw = self.output_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return image[:, i:i + th, j:j + tw]


# Original CNN architectures
class MyNetwork(nn.Module):
    def __init__(self, size=3, drop_rate=0.5):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=size)
        self.conv2_drop = nn.Dropout(drop_rate)
        self._to_linear = None

        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.conv2,
            nn.ReLU(),
        )
        self._get_conv_output()
        self.fc1 = nn.Linear(self._to_linear, 50)
        self.fc2 = nn.Linear(50, 2)

    def _get_conv_output(self):
        x = torch.randn(1, 3, 64, 64)
        x = self.convs(x)
        self._to_linear = x.numel() // x.shape[0]

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyNetwork2(nn.Module):
    def __init__(self, size=3, drop_rate=0.5):
        super(MyNetwork2, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=size)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=size)
        self.conv_drop = nn.Dropout(drop_rate)
        self._to_linear = None

        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self._get_conv_output()
        self.fc1 = nn.Linear(self._to_linear, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def _get_conv_output(self):
        x = torch.randn(1, 3, 64, 64)
        x = self.convs(x)
        self._to_linear = x.numel() // x.shape[0]

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, epoch, train_losses, train_counter):
    model.train()
    criterion = AngularLoss()
    total_loss = 0
    total_samples = 0
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        total_samples += batch_size

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size

        if batch_index % 50 == 0:
            train_losses.append(loss.item())
            train_counter.append((batch_index * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            print(f'Train Epoch: {epoch} [{batch_index * batch_size}/{len(train_loader.dataset)} '
                  f'({100. * batch_index / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} degrees')

    average_loss = total_loss / total_samples
    print(f'Train Epoch: {epoch} \tAverage Train Loss: {average_loss:.6f} degrees')


def validate(model, device, valid_loader, valid_losses, epoch, valid_counter, train_loader):
    model.eval()
    test_loss = 0
    criterion = AngularLoss(reduction='sum')
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    test_loss = test_loss / len(valid_loader.dataset)
    valid_losses.append(test_loss)
    valid_counter.append(epoch * len(train_loader.dataset))
    return test_loss


def main():
    parser = argparse.ArgumentParser(description='Train a neural network on spectral ratio data with separate train and validation sets.')
    parser.add_argument('train_csv_file', type=str, help='Path to the training CSV file')
    parser.add_argument('train_img_dir', type=str, help='Directory containing the training images')
    parser.add_argument('valid_csv_file', type=str, help='Path to the validation CSV file')
    parser.add_argument('valid_img_dir', type=str, help='Directory containing the validation images')
    parser.add_argument('model', type=int, choices=[1, 2], help='Choose the model: 1 for MyNetwork, 2 for MyNetwork2')
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose the model
    if args.model == 1:
        model = MyNetwork().to(device)
        print("Using MyNetwork (Model 1)")
    else:
        model = MyNetwork2().to(device)
        print("Using MyNetwork2 (Model 2)")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Normalization values (can adjust if needed)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        RandomCropTensor((64, 64)),
        transforms.Normalize(mean=mean, std=std),
    ])

    valid_transform = transforms.Compose([
        CenterCropTensor((64, 64)),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Create training and validation datasets
    train_dataset = helper_function.SpectralRatioDataset(args.train_csv_file, args.train_img_dir, transform=train_transform)
    valid_dataset = helper_function.SpectralRatioDataset(args.valid_csv_file, args.valid_img_dir, transform=valid_transform)

    # Data loaders
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    train_counter = []
    valid_losses = []
    valid_counter = []
    max_epochs = 30

    best_valid_loss = float('inf')
    best_epoch = 0

    # Initial validation
    initial_loss = validate(model, device, valid_loader, valid_losses, 0, valid_counter, train_loader)
    if initial_loss < best_valid_loss:
        best_valid_loss = initial_loss
        best_epoch = 0
        model_path = 'model_epoch_0.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Best model saved at epoch {best_epoch} to {model_path}')
    print(f'Validation set: Average angular loss: {initial_loss:.4f} degrees, best average angular loss: {best_valid_loss:.4f} degrees\n')

    # Training loop
    for epoch in range(1, max_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, train_losses, train_counter)
        valid_loss = validate(model, device, valid_loader, valid_losses, epoch, valid_counter, train_loader)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
        model_path = f'model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Validation set: Average angular loss: {valid_loss:.4f} degrees, '
              f'best average angular loss: {best_valid_loss:.4f} degrees\n')

    # Save final model
    torch.save(model.state_dict(), 'trained_model_state.pth')
    print(f'Best model was at epoch {best_epoch} with an angular loss of {best_valid_loss:.4f} degrees')


if __name__ == "__main__":
    main()


