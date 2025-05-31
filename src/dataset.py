import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, num_classes=19):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)
    def map_labels_to_train_ids(self, mask: np.ndarray) -> np.ndarray:
        """
        Remap Cityscapes original label‐IDs to train‐IDs (0–18).
        All other pixels → 255 (ignore).
        """
        mapped = np.ones_like(mask, dtype=np.uint8) * 255
        for orig_id, train_id in config.LABEL_ID_TO_TRAIN_ID.items():
            mapped[mask == orig_id] = train_id
        return mapped
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            # For Cityscapes, masks are usually single-channel with label IDs
            mask = np.array(Image.open(mask_path)) # Keep as is, no .convert("L") initially
                                                   # Preprocessing for specific class IDs should happen later
                                                   # or ensure your masks are already in the correct format.
            mask = self.map_labels_to_train_ids(mask)
        except FileNotFoundError:
            print(f"Error: File not found. Image: {img_path} or Mask: {mask_path}")
            # Return dummy data or raise error, depending on how you want to handle this
            return torch.randn((3, 256, 256)), torch.zeros((256, 256), dtype=torch.long)
        except Exception as e:
            print(f"Error loading image/mask: {img_path}, {mask_path}. Error: {e}")
            return torch.randn((3, 256, 256)), torch.zeros((256, 256), dtype=torch.long)


        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure mask is of the correct type and shape
        # For CrossEntropyLoss, mask should be LongTensor with class indices [H, W]
        # For BCEWithLogitsLoss (binary), mask should be FloatTensor [H, W] or [1, H, W]
        
        # If using Cityscapes, you might need to map original label IDs to training IDs (0 to N-1)
        # This is a placeholder for such mapping. You'll need to implement map_labels_to_train_ids
        # mask = self.map_labels_to_train_ids(mask) # Example

        if self.num_classes > 1: # Multi-class
            mask = mask.long()
        else: # Binary
            mask = mask.float().unsqueeze(0) # Add channel dim: [1, H, W]

        return image, mask

    # def map_labels_to_train_ids(self, mask):
    #     # Placeholder: Implement your Cityscapes label mapping here
    #     # e.g., map original IDs to a contiguous range 0-(num_classes-1)
    #     # and map ignored labels to a specific ignore_index (e.g., 255)
    #     # For example:
    #     # train_id_map = {7: 0, 8: 1, ..., 255: 255} # map original id 7 to train id 0, etc.
    #     # new_mask = np.full_like(mask, fill_value=self.ignore_index if hasattr(self, 'ignore_index') else 255)
    #     # for original_id, train_id in train_id_map.items():
    #     #    new_mask[mask == original_id] = train_id
    #     # return new_mask
    #     return mask # Return as is if no mapping is done here

def get_transforms(height, width, is_train=True):
    if is_train:
        transform = A.Compose([
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(height, width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return transform

def get_image_mask_paths(image_dir, mask_dir):
    """
    Generates lists of corresponding image and mask paths for Cityscapes.
    It expects image_dir and mask_dir to point to directories that may contain
    city subfolders (e.g., 'aachen', 'bochum') or directly contain the images/masks
    if pointed to a specific city subfolder.
    """
    image_paths = []
    mask_paths = []

    # Determine the mask type suffix based on the mask_dir path
    # This helps select between gtFine and gtCoarse if both might be present
    # or if the naming convention is consistent.
    mask_type_suffix = "_gtFine_labelIds.png" # Default to gtFine
    if "gtcoarse" in mask_dir.lower() and "gtfine" not in mask_dir.lower(): # Check specifically for gtCoarse
        mask_type_suffix = "_gtCoarse_labelIds.png"
    elif "gtfine" in mask_dir.lower(): # Explicitly check for gtFine
        mask_type_suffix = "_gtFine_labelIds.png"
    # Add more sophisticated logic here if needed, or pass as a parameter

    print(f"Using mask suffix: {mask_type_suffix}")

    for root, _, files in os.walk(image_dir):
        for file_name in sorted(files):
            if file_name.lower().endswith('_leftimg8bit.png'): # Cityscapes image naming
                img_path = os.path.join(root, file_name)
                
                # Construct the corresponding mask file name
                base_name = file_name.replace('_leftImg8bit.png', '')
                mask_file_name = base_name + mask_type_suffix
                
                # Determine the relative path from the base image_dir to the current root
                # This helps find the corresponding city subfolder in the mask_dir
                relative_dir_path = os.path.relpath(root, image_dir)
                if relative_dir_path == '.': # In case image_dir is the direct parent
                    relative_dir_path = ''
                
                potential_mask_path = os.path.join(mask_dir, relative_dir_path, mask_file_name)
                
                if os.path.exists(potential_mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(potential_mask_path)
                else:
                    print(f"Warning: Mask not found for image {img_path}. Expected at {potential_mask_path}")

    if not image_paths:
        print(f"Warning: No images found in {image_dir} ending with '_leftImg8bit.png'.")
    if not mask_paths:
        print(f"Warning: No mask files found. Check MASK_DIR, naming conventions ('{mask_type_suffix}'), and paths.")
    
    if image_paths and mask_paths:
        print(f"Found {len(image_paths)} image(s) and {len(mask_paths)} corresponding mask(s).")
        if len(image_paths) != len(mask_paths):
            print(f"CRITICAL WARNING: Mismatch in number of images and masks found. This will lead to errors.")
            # You might want to raise an error here or handle it more gracefully depending on requirements
    
    return image_paths, mask_paths

if __name__ == '__main__':
    # Example usage:
    image_paths, mask_paths = get_image_mask_paths(config.IMAGE_DIR, config.MASK_DIR)
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")

    if not image_paths or not mask_paths:
        print("No images or masks found. Please check your data directory and naming conventions.")
        print("Expected structure: data/images/your_image.png, data/masks/your_image_mask.png")
    else:
        train_transform = get_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, is_train=True)
        dataset = SegmentationDataset(image_paths, mask_paths, transform=train_transform)

        if len(dataset) > 0:
            img, msk = dataset[0]
            print("Image shape:", img.shape)
            print("Mask shape:", msk.shape)
            print("Mask unique values:", torch.unique(msk))
        else:
            print("Dataset is empty after attempting to load paths.")
