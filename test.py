import torch
import torchvision.transforms as transforms
import argparse
import numpy as np
from dataloader import get_test_loader
from model import CNNExperiment
import matplotlib.pyplot as plt
from PIL import Image
import os
import time  # Import the time module


def calculate_statistics(predictions, targets):
    distances = np.linalg.norm(predictions - targets, axis=1)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    return {
        'min': np.min(distances),
        'mean': np.mean(distances),
        'max': np.max(distances),
        'std': np.std(distances),
        'mae': mae,
        'rmse': rmse
    }


def visualize_results(image, true_coord, pred_coord, index, output_dir):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
    plt.plot(true_coord[0], true_coord[1], 'go', markersize=15, label='True')
    plt.plot(pred_coord[0], pred_coord[1], 'ro', markersize=15, label='Predicted')
    plt.legend()
    plt.title(f'Sample {index}')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'sample_{index}.png'))
    plt.close()


def main():
    print('Running main ...')

    # Read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, required=True, help='Parameter file (.pth)')
    argParser.add_argument('-b', metavar='batch size', type=int, default=32, help='Batch size [32]')
    argParser.add_argument('-t', metavar='transform', type=str, default='flip',
                           help='Transform type [flip] or [saturate]')

    args = argParser.parse_args()

    if args.s is not None:
        save_file = args.s
    if args.b is not None:
        batch_size = args.b
    if args.t is not None:
        transform = args.t

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\t\tUsing device:', device)

    # Replace with your actual dataset
    test_loader = get_test_loader(batch_size=batch_size, transform_type=transform)

    # Initialize the model
    model = CNNExperiment()  # Replace with your model initialization if needed
    model.load_state_dict(torch.load(save_file, weights_only=True))
    model.to(device)
    model.eval()

    predictions = []
    targets = []

    total_time = 0  # Variable to accumulate total time
    total_images = 0  # Variable to keep track of the total number of images

    with torch.no_grad():
        for data in test_loader:
            imgs, coords = data  # Assuming the dataset returns images and target coordinates
            imgs = imgs.to(device)
            coords = coords.to(device)

            # Start the timer
            start_time = time.time()

            outputs = model(imgs)

            # End the timer and compute the time taken for this batch
            batch_time = time.time() - start_time
            total_time += batch_time
            total_images += imgs.size(0)

            predictions.append(outputs.cpu().numpy())
            targets.append(coords.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # Calculate statistics
    stats = calculate_statistics(predictions, targets)

    print('Localization Accuracy Statistics:')
    print(f"Minimum Distance: {stats['min']:.2f}")
    print(f"Mean Distance: {stats['mean']:.2f}")
    print(f"Maximum Distance: {stats['max']:.2f}")
    print(f"Standard Deviation: {stats['std']:.2f}")
    print(f"Mean Absolute Error: {stats['mae']:.2f}")
    print(f"Root Mean Square Error: {stats['rmse']:.2f}")

    # Calculate and print the average time per image in milliseconds
    avg_time_per_image = (total_time / total_images) * 1000  # Convert to milliseconds
    print(f"Average time per image: {avg_time_per_image:f} ms")


if __name__ == '__main__':
    main()
