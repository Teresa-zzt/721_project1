import os
from collections import Counter
import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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

def data_preprocess(batch_size=32):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor()
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = datasets.ImageFolder('top8_lfw', transform=transform_train)
    class_names = full_dataset.classes

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Override val_ds transform
    val_ds.dataset.transform = transform_val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, class_names


