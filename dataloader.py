import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ast import literal_eval

# Specify the directories for images and labels
img_dir = './oxford-iiit-pet-noses/images-original/images'
train_label_file = './oxford-iiit-pet-noses/train_noses.txt'
test_label_file = './oxford-iiit-pet-noses/test_noses.txt'


class PetNoseDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self._read_labels(label_file)

    def _read_labels(self, label_file):
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    img_name, nose_coords_str = line.split(',', 1)
                    nose_coords_str = nose_coords_str.strip('"')
                    nose_coords = literal_eval(nose_coords_str)
                    labels.append((img_name, nose_coords))
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing line: {line}. Error: {e}")
                    continue
        print(f"Total labels loaded: {len(labels)}")
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels[idx][0])

        try:
            image = Image.open(img_name).convert("RGB")
        except (IOError, FileNotFoundError) as e:
            return None  # Return None if the image fails to load

        coordinates = self.labels[idx][1]

        # Get the original image dimensions
        orig_width, orig_height = image.size
        # Scale the coordinates based on the resizing
        x_orig, y_orig = coordinates
        x_new = x_orig * (227 / orig_width)
        y_new = y_orig * (227 / orig_height)
        scaled_coordinates = [x_new, y_new]

        if self.transform:
            image = self.transform(image)

        # Convert coordinates to tensor
        scaled_coordinates = torch.tensor(scaled_coordinates, dtype=torch.float32)

        return image, scaled_coordinates


def get_transform(transform_type=None):
    base_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if transform_type == 'horizontal_flip' or transform_type == 'all':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            base_transform
        ])
    elif transform_type == 'saturation' or transform_type == 'all':
        return transforms.Compose([
            transforms.ColorJitter(saturation=0.5),
            base_transform
        ])
    else:
        return base_transform


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise ValueError("All images in this batch failed to load.")
    images, coordinates = zip(*batch)
    return torch.stack(images, 0), torch.stack(coordinates, 0)


def get_train_loader(batch_size=32, transform_type=None):
    train_transform = get_transform(transform_type)
    train_dataset = PetNoseDataset(img_dir=img_dir, label_file=train_label_file, transform=train_transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)


def get_test_loader(batch_size=32, transform_type=None):
    test_transform = get_transform(transform_type)  # No additional transforms for test data
    test_dataset = PetNoseDataset(img_dir=img_dir, label_file=test_label_file, transform=test_transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


def reality_check(data_loader, num_samples=5):
    """
    A routine to check the data loading process.
    Prints image tensor values and corresponding labels.
    Optionally visualizes images and their ground truth labels.
    """
    for i, (images, labels) in enumerate(data_loader):
        if images is None:
            continue

        print(f"Batch {i + 1}:")
        print(f"Image batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")

        # Print image tensor values and corresponding labels
        for j in range(min(num_samples, images.size(0))):
            print(f"Image {j + 1} values: {images[j].numpy()}")
            print(f"Label {j + 1}: {labels[j].numpy()}")

        # Visualize the images and labels
        for j in range(min(num_samples, images.size(0))):
            plt.figure(figsize=(5, 5))
            plt.imshow(images[j].permute(1, 2, 0).numpy())  # Convert from CxHxW to HxWxC
            plt.scatter(labels[j][0].item(), labels[j][1].item(), color='red', label='Ground Truth')
            plt.title(f"Sample {j + 1} - Batch {i + 1}")
            plt.axis('off')
            plt.legend()
            plt.show()

        if i >= num_samples - 1:  # Limit to a few batches
            break


# Example usage
if __name__ == '__main__':
    train_loader = get_train_loader(batch_size=32)
    reality_check(train_loader, num_samples=3)


