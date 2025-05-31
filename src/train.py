import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np

import config
from dataset import SegmentationDataset, get_transforms, get_image_mask_paths
from model import UNet # Or your custom model, or SOTA model function
# from src.model import get_sota_model # If using segmentation_models_pytorch
from utils import calculate_miou_batch, plot_training_curves, visualize_predictions

def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    epoch_loss = 0.0
    epoch_miou = 0.0
    for images, masks in dataloader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long if num_classes > 1 else torch.float32)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks.squeeze(1) if num_classes > 1 and len(masks.shape) == 4 else masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        miou = calculate_miou_batch(outputs, masks, num_classes)
        epoch_miou += miou

    return epoch_loss / len(dataloader), epoch_miou / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device, num_classes, epoch, fold, visualize=False):
    model.eval()
    epoch_loss = 0.0
    epoch_miou = 0.0
    
    # For visualization
    all_images, all_true_masks, all_pred_masks = [], [], [] 
    visualized_this_epoch = False

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long if num_classes > 1 else torch.float32)

            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1) if num_classes > 1 and len(masks.shape) == 4 else masks)
            epoch_loss += loss.item()

            miou = calculate_miou_batch(outputs, masks, num_classes)
            epoch_miou += miou

            # Store some samples for visualization
            if visualize and not visualized_this_epoch and i == 0: # Visualize first batch of an epoch
                if len(all_images) < config.VISUALIZATION_COUNT:
                    all_images.append(images.cpu())
                    all_true_masks.append(masks.cpu())
                    all_pred_masks.append(outputs.cpu())
    
    if visualize and all_images:
        # Concatenate along the batch dimension
        vis_images = torch.cat(all_images, dim=0)
        vis_true_masks = torch.cat(all_true_masks, dim=0)
        vis_pred_masks = torch.cat(all_pred_masks, dim=0)
        
        save_path = os.path.join(config.RESULTS_SAVE_PATH, f'fold_{fold+1}', 'visualizations')
        visualize_predictions(vis_images, vis_true_masks, vis_pred_masks,
                              num_samples=config.VISUALIZATION_COUNT,
                              fold=fold, epoch=epoch,
                              save_dir=save_path,
                              num_classes=num_classes)
        visualized_this_epoch = True # Ensure visualization happens once per validation epoch if enabled

    return epoch_loss / len(dataloader), epoch_miou / len(dataloader)

def main():
    # Create necessary directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.RESULTS_SAVE_PATH, exist_ok=True)

    # Load dataset paths
    all_image_paths, all_mask_paths = get_image_mask_paths(config.IMAGE_DIR, config.MASK_DIR)

    if not all_image_paths or not all_mask_paths:
        print("No data found. Please check config.py and your data directories.")
        return
    
    # Ensure they are numpy arrays for KFold indexing
    all_image_paths = np.array(all_image_paths)
    all_mask_paths = np.array(all_mask_paths)

    kf = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_image_paths)):
        print(f"--- Fold {fold+1}/{config.NUM_FOLDS} ---")

        train_image_paths, val_image_paths = all_image_paths[train_idx], all_image_paths[val_idx]
        train_mask_paths, val_mask_paths = all_mask_paths[train_idx], all_mask_paths[val_idx]

        # Datasets and DataLoaders
        train_transform = get_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, is_train=True)
        val_transform = get_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, is_train=False)

        train_dataset = SegmentationDataset(train_image_paths.tolist(), train_mask_paths.tolist(), transform=train_transform)
        val_dataset = SegmentationDataset(val_image_paths.tolist(), val_mask_paths.tolist(), transform=val_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size   = config.BATCH_SIZE,
            shuffle      = True,
            num_workers  = config.NUM_WORKERS,
            pin_memory   = config.PIN_MEMORY,
        )
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        # Model
        device = torch.device(config.DEVICE)
        # Choose your model. Example: Basic U-Net
        model = UNet(n_class=config.NUM_CLASSES).to(device)
        # Or use a SOTA model from segmentation_models_pytorch (ensure it's installed and uncommented in model.py)
        # model = get_sota_model(
        #     model_name='UnetPlusPlus', # Example
        #     encoder_name=config.ENCODER,
        #     encoder_weights=config.ENCODER_WEIGHTS,
        #     num_classes=config.NUM_CLASSES,
        #     activation=None # For BCEWithLogitsLoss or CrossEntropyLoss
        # ).to(device)

        # Loss function
        if config.NUM_CLASSES == 1:
            # Binary segmentation (output is 1 channel, target is 1 channel)
            criterion = nn.BCEWithLogitsLoss()
        else:
            # Multi-class segmentation (output is C channels, target is 1 channel with class indices)
            criterion = nn.CrossEntropyLoss(ignore_index=255)

        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

        best_val_miou = -1.0
        history = {'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': []}

        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_miou = train_one_epoch(model, train_loader, criterion, optimizer, device, config.NUM_CLASSES)
            val_loss, val_miou = validate_one_epoch(model, val_loader, criterion, device, config.NUM_CLASSES, epoch + 1, fold, visualize=(epoch % 5 == 0 or epoch == config.NUM_EPOCHS -1) ) # Visualize every 5 epochs and last epoch

            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} -> "
                  f"Train Loss: {train_loss:.4f}, Train MIoU: {train_miou:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val MIoU: {val_miou:.4f}")

            history['train_loss'].append(train_loss)
            history['train_miou'].append(train_miou)
            history['val_loss'].append(val_loss)
            history['val_miou'].append(val_miou)

            # scheduler.step(val_loss)

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                model_fold_path = os.path.join(config.MODEL_SAVE_PATH, f'best_model_fold_{fold+1}.pth')
                torch.save(model.state_dict(), model_fold_path)
                print(f"Saved best model for fold {fold+1} to {model_fold_path} (MIoU: {best_val_miou:.4f})")
        
        fold_results.append(best_val_miou)
        plot_training_curves(history, fold, os.path.join(config.RESULTS_SAVE_PATH, f'fold_{fold+1}'))
        print(f"Best Validation MIoU for Fold {fold+1}: {best_val_miou:.4f}")

    print("\n--- Cross-Validation Summary ---")
    for i, miou in enumerate(fold_results):
        print(f"Fold {i+1} Best MIoU: {miou:.4f}")
    print(f"Average MIoU across {config.NUM_FOLDS} folds: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")

if __name__ == '__main__':
    main()
