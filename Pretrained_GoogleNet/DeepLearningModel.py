
# version of GoogleNet + bn
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim
import helper_function
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
from torchvision.models.googlenet import BasicConv2d

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

        # Compute angle in radians and convert to degrees
        angle_rad = torch.acos(dot_product)
        angle_deg = angle_rad * (180.0 / np.pi)

        if self.reduction == 'mean':
            return torch.mean(angle_deg)
        elif self.reduction == 'sum':
            return torch.sum(angle_deg)
        else:
            return angle_deg

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

# pretrained googleNet
def modify_googlenet_for_smaller_images():
    googlenet = models.googlenet(pretrained=True)
    
    # Modify the first convolutional layer to accept smaller images
    googlenet.conv1.conv = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64)
    )

    # Collect names of Conv2d modules for batch normalization
    conv2d_modules = []
    for name, module in googlenet.named_modules():
        if isinstance(module, nn.Conv2d):
            conv2d_modules.append((name, module))

    # Add BatchNorm2d after each Conv2d
    for name, module in conv2d_modules:
        parent_module = googlenet
        name_parts = name.split('.')
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        
        # Replace the Conv2d module with a Sequential block including BatchNorm2d
        setattr(parent_module, name_parts[-1], nn.Sequential(
            module,
            nn.BatchNorm2d(module.out_channels)
        ))

    # Remove or adjust the first maxpool layer
    googlenet.maxpool1 = nn.Identity()
    # Remove or adjust the second maxpool layer
    googlenet.maxpool2 = nn.Identity()
    # Remove or adjust the third maxpool layer
    googlenet.maxpool3 = nn.Identity()
    # Adjust the fully connected layer
    num_ftrs = googlenet.fc.in_features
    googlenet.fc = nn.Linear(num_ftrs, 2)  # Output azimuth and elevation angles

    return googlenet


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

        total_loss += loss.item() * batch_size  # Multiply by batch size to get total loss over all samples

        if batch_index % 50 == 0:
            train_losses.append(loss.item())
            train_counter.append((batch_index * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            print(f'Train Epoch: {epoch} [{batch_index * batch_size}/{len(train_loader.dataset)} '
                  f'({100. * batch_index / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} degrees')

    # After the epoch, compute average training loss
    average_loss = total_loss / total_samples
    print(f'Train Epoch: {epoch} \tAverage Train Loss: {average_loss:.6f} degrees')


def test(model, device, valid_loader, valid_losses, epoch, valid_counter, train_loader):
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
    return test_loss  # Return the average validation loss


def main(argv):
    if len(argv) != 5:
        print("Usage: python your_script.py <train_csv_file> <train_img_dir> <valid_csv_file> <valid_img_dir>")
        sys.exit(1)

    # Parse command-line arguments
    train_csv_file = argv[1]
    train_img_dir = argv[2]
    valid_csv_file = argv[3]
    valid_img_dir = argv[4]

    torch.manual_seed(42)
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modify_googlenet_for_smaller_images().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Mean and std for normalization (using ImageNet values)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define transformations
    train_transform = transforms.Compose([
        RandomCropTensor((64, 64)),
        transforms.Normalize(mean=mean, std=std),
    ])

    valid_transform = transforms.Compose([
        CenterCropTensor((64, 64)),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Create training and validation datasets with transforms
    train_dataset = helper_function.SpectralRatioDataset(train_csv_file, train_img_dir, transform=train_transform)
    valid_dataset = helper_function.SpectralRatioDataset(valid_csv_file, valid_img_dir, transform=valid_transform)

    # Data Loaders
    batch_size = 10  # Adjust as needed
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    train_counter = []
    valid_losses = []
    valid_counter = []
    max_epochs = 50

    best_valid_loss = float('inf')
    best_epoch = 0  # To keep track of the epoch with the best model

    # Initial validation
    valid_loss = test(model, device, valid_loader, valid_losses, 0, valid_counter, train_loader)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = 0
        # Save the best model
        model_path = 'used_for_paper/model0.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Best model saved at epoch {best_epoch} to {model_path}')
    print(f'Validation set: Average angular loss: {valid_loss:.4f} degrees, best average angular loss: {best_valid_loss:.4f} degrees')
    print()

    for epoch in range(1, max_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, train_losses, train_counter)
        valid_loss = test(model, device, valid_loader, valid_losses, epoch, valid_counter, train_loader)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
        # Save the best model
        model_path = f'used_for_paper/model{epoch}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Validation set: Average angular loss: {valid_loss:.4f} degrees, best average angular loss: {best_valid_loss:.4f} degrees')
        print()

    # Optionally, save the final model
    torch.save(model.state_dict(), 'trained_googlenet_state.pth')

if __name__ == "__main__":
    main(sys.argv)

