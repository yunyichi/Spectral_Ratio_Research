import torch
import helper_function
import torchvision.transforms as transforms
from DeepLearningModel import MyNetwork, test, CenterCropTensor
import argparse
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test trained neural networks on spectral ratio data.')
    parser.add_argument('csv_file', type=str,
                        help='Path to the CSV file containing labels')
    parser.add_argument('img_dir', type=str,
                        help='Directory containing the images')
    parser.add_argument('model_dir', type=str,
                        help='Directory containing the trained model state files')
    args = parser.parse_args()

    # Set the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the test data transformations
    test_transform = transforms.Compose([
        CenterCropTensor((64, 64)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # Load the full dataset
    csv_file = args.csv_file
    img_dir = args.img_dir
    full_dataset = helper_function.SpectralRatioDataset(csv_file, img_dir, transform=test_transform)

    # Use all data for testing
    test_indices = list(range(len(full_dataset)))

    # Create the test dataset and loader
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Get list of model files in the specified directory
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pth')]
    model_files.sort()  # Optional: sort the model files

    # Iterate over each model file
    for model_file in model_files:
        model_path = os.path.join(args.model_dir, model_file)
        print(f"\nTesting model: {model_file}")

        # Initialize the model
        model = MyNetwork().to(device)

        # Load the trained model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set the model to evaluation mode

        # Prepare empty lists for losses and counters
        test_losses = []
        test_counter = []

        # Pass test_loader as train_loader to avoid AttributeError
        test(model, device, test_loader, test_losses, epoch=1, valid_counter=test_counter, train_loader=test_loader)

        # Print the test loss for this model
        if test_losses:
            avg_test_loss = test_losses[-1]
            print(f"Model {model_file} - Test Loss: {avg_test_loss:.4f} degrees")
        else:
            print(f"No test loss recorded for model {model_file}")

if __name__ == "__main__":
    main()
