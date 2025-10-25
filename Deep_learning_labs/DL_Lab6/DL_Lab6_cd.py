# Loading custom dataset - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html 

import os
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def load_custom_dataset(label_csv_path, image_directory, transform=None):
    label_data = pd.read_csv(label_csv_path)
    image_names = label_data.iloc[:, 0].values
    image_labels = label_data.iloc[:, 1].values

    images = []
    labels = []

    for image_name, label in zip(image_names, image_labels):
        image_path = os.path.join(image_directory, image_name)
        image = read_image(image_path)  # Load image tensor (C × H × W)
        if transform:
            image = transform(image)
        images.append(image)
        labels.append(label)

    # Stack all images into one tensor and convert labels to tensor
    image_tensors = torch.stack(images)
    label_tensors = torch.tensor(labels, dtype=torch.long)

    return image_tensors, label_tensors


def main():
    # Define file paths
    csv_path = './data/labels.csv'
    image_folder = './data/images'

    # Define transform
    image_transform = ToTensor()

    # Load dataset 
    image_tensors, label_tensors = load_custom_dataset(
        label_csv_path=csv_path,
        image_directory=image_folder,
        transform=image_transform
    )

    print(f"Loaded {len(image_tensors)} images.")

    # Create DataLoader using TensorDataset
    dataset = TensorDataset(image_tensors, label_tensors)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Fetch one batch 
    image_batch, label_batch = next(iter(data_loader))
    print(f"Image batch shape: {image_batch.size()}")
    print(f"Label batch shape: {label_batch.size()}")

    # Display first image from batch 
    first_image = image_batch[0].permute(1, 2, 0) 
    first_label = label_batch[0].item()

    plt.imshow(first_image)
    plt.title(f"Label: {first_label}")
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main()
