import os
from collections import Counter
import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

def top8classes_finding():
# Path to the LFW folder
    lfw_path = 'lfw'  # Adjust this if needed
    new_dir = 'top8_lfw'
    import os
    if os.path.isdir(new_dir):
        print("Folder 'top8_lfw' exists.")
    else:      
        # Count number of images in each class folder
        class_image_counts = {}

        for class_name in os.listdir(lfw_path):
            class_dir = os.path.join(lfw_path, class_name)
            if os.path.isdir(class_dir):
                num_images = len([
                    f for f in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                class_image_counts[class_name] = num_images

        # Sort and pick top 8 classes
        top_8_classes = sorted(class_image_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        top_8_class_names = [name for name, _ in top_8_classes]

        print("Top 8 classes:", top_8_class_names)

        os.makedirs(new_dir, exist_ok=True)

        for class_name in top_8_class_names:
            src = os.path.join(lfw_path, class_name)
            dst = os.path.join(new_dir, class_name)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
            print("Create a new folder that only contain the top 8 classes")

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def data_preprocess(batch_size=32):
    transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load full dataset with train transforms (override later for val/test)
    full_dataset = datasets.ImageFolder('top8_lfw', transform=transform_train)
    class_names = full_dataset.classes


    # train_size = int(0.7 * len(full_dataset))
    # val_size = int(0.15 * len(full_dataset))
    # test_size = len(full_dataset) - train_size - val_size
    # seed = 42
    # generator = torch.Generator().manual_seed(seed)
    # train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    targets = [label for _, label in full_dataset.samples]

    train_idx, temp_idx = train_test_split(
        list(range(len(targets))), test_size=0.3, stratify=targets, random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[targets[i] for i in temp_idx], random_state=42
    )

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, class_names


