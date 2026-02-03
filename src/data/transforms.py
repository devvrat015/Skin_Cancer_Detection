from torchvision import transforms

# Training transforms (with augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # ResNet-50 input size
   # transforms.RandomHorizontalFlip(),       # augmentation
    #transforms.RandomRotation(10),            # augmentation
    transforms.ToTensor(),                    # convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],           # ImageNet mean
        std=[0.229, 0.224, 0.225]              # ImageNet std
    )
])


# Validation transforms (NO augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
