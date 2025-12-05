# --- dataset_hf.py ---
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset 
from datasets import load_dataset
from PIL import Image

# Import configurations
from config import HF_DATASET_NAME, BATCH_SIZE_TRAIN, BATCH_SIZE_EVAL, DEVICE

# --- 1. Custom PyTorch Dataset for Hugging Face ---

class HuggingFaceCelebADataset(Dataset):
    """
    Wraps the Hugging Face CelebA dataset split for PyTorch DataLoader.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        
        # Identify the 40 attribute names by skipping 'image' and 'celeb_id'
        all_features = list(hf_dataset.features.keys())
        self.attr_names = all_features[2:]

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        
        # Load image (PIL.Image)
        img: Image.Image = item['image']
        
        # Extract labels (40 attributes)
        # item['attribute_name'] returns True/False
        labels_list = [item[name] for name in self.attr_names]
        
        # Convert True/False to 1.0/0.0 float32 tensor
        target = torch.tensor(labels_list, dtype=torch.float32)

        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

# --- 2. Transformation Function ---

def get_transforms(model_name):
    """
    Defines and returns standard image transformations, adapted for ResNet/ViT input size.
    ViT often benefits from slightly larger crops or higher resolution.
    """
    # Standard ImageNet normalization values
    standard_mean = [0.485, 0.456, 0.406]
    standard_std = [0.229, 0.224, 0.225]
    
    # Use 224x224, standard for ResNet and ViT-16/b
    IMG_SIZE = 224
    
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=standard_mean, std=standard_std)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=standard_mean, std=standard_std)
    ])
    
    return train_transforms, val_test_transforms

# --- 3. Main Initialization Function ---

def initialize_data_loaders(model_name=None):
    """Loads the Hugging Face dataset and returns DataLoaders."""
    
    print(f"Loading CelebA dataset from Hugging Face: {HF_DATASET_NAME}")
    
    # Load the standard splits: train, validation, test
    ds = load_dataset(HF_DATASET_NAME)
    
    # 1. Get transforms
    train_transforms, val_test_transforms = get_transforms(model_name)

    # 2. Create datasets
    train_dataset = HuggingFaceCelebADataset(ds['train'], transform=train_transforms)
    val_dataset = HuggingFaceCelebADataset(ds['valid'], transform=val_test_transforms)
    test_dataset = HuggingFaceCelebADataset(ds['test'], transform=val_test_transforms)
    
    # Extract attribute names for metrics logging
    attribute_names = train_dataset.attr_names

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=4
    )

    print("Data Loaders Initialized Successfully.")
    print(f"Train Samples: {len(train_dataset)}, Val Samples: {len(val_dataset)}, Test Samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, attribute_names