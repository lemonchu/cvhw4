import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import config

def calculate_iou(pred_mask, true_mask, smooth=1e-6):
    """Calculates Intersection over Union (IoU) for a single prediction.
    Args:
        pred_mask (torch.Tensor): Predicted mask (binary, 0 or 1), shape [H, W] or [1, H, W].
        true_mask (torch.Tensor): Ground truth mask (binary, 0 or 1), shape [H, W] or [1, H, W].
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        float: IoU score.
    """
    pred_mask = pred_mask.squeeze().byte() # Ensure it's [H, W] and boolean/byte
    true_mask = true_mask.squeeze().byte()

    intersection = (pred_mask & true_mask).float().sum()
    union = (pred_mask | true_mask).float().sum()

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calculate_miou_batch(preds, targets, num_classes, smooth=1e-6):
    """Calculates Mean Intersection over Union (MIoU) for a batch of predictions.
    Args:
        preds (torch.Tensor): Batch of predicted masks. Shape (N, C, H, W) for multi-class (logits/probs)
                              or (N, 1, H, W) for binary (logits/probs).
        targets (torch.Tensor): Batch of ground truth masks. Shape (N, H, W) for multi-class (class indices)
                                or (N, 1, H, W) for binary (0 or 1).
        num_classes (int): Number of classes.
        smooth (float): Smoothing factor.
    Returns:
        float: Mean IoU score over the batch.
    """
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    if num_classes == 1: # Binary segmentation
        # Assuming preds are logits, apply sigmoid and threshold
        preds_binary = (torch.sigmoid(preds) > 0.5).squeeze(1) # (N, H, W)
        targets_binary = targets.squeeze(1) # (N, H, W)
        batch_iou = []
        for i in range(preds_binary.shape[0]):
            batch_iou.append(calculate_iou(preds_binary[i], targets_binary[i], smooth))
        return np.nanmean(batch_iou) if batch_iou else 0.0

    else: # Multi-class segmentation
        # Assuming preds are logits, get class predictions by argmax
        preds_classes = torch.argmax(preds, dim=1) # (N, H, W)
        # targets are already class indices (N, H, W)

        iou_per_class_batch = []
        for i in range(preds_classes.shape[0]): # Iterate over batch
            pred_img = preds_classes[i]
            target_img = targets[i]
            iou_per_class_img = []
            for cls in range(num_classes):
                pred_inds = (pred_img == cls)
                target_inds = (target_img == cls)

                intersection = (pred_inds & target_inds).float().sum()
                union = (pred_inds | target_inds).float().sum()

                if union == 0:
                    # If there is no ground truth and no prediction for this class, IoU is 1
                    # If there is no ground truth but a prediction, IoU is 0
                    # If there is ground truth but no prediction, IoU is 0
                    # This typically means the class is not present in this image, or not predicted.
                    # Some implementations might skip this class for mIoU calculation if not in GT.
                    # Here, we add NaN and handle it with nanmean later, or you can assign a value (e.g., 1 if intersection is also 0)
                    iou = np.nan if intersection == 0 else 0.0
                else:
                    iou = (intersection + smooth) / (union + smooth)
                iou_per_class_img.append(iou.item() if isinstance(iou, torch.Tensor) else iou)

            # Mean IoU for the current image (excluding NaNs for classes not present in GT)
            img_miou = np.nanmean([iou for iou in iou_per_class_img if not np.isnan(iou)])
            if not np.isnan(img_miou):
                 iou_per_class_batch.append(img_miou)

        return np.nanmean(iou_per_class_batch) if iou_per_class_batch else 0.0

def plot_training_curves(history, fold, save_dir):
    """Plots training and validation loss and mIoU curves."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'Fold {fold+1} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_miou'], 'bo-', label='Training MIoU')
    plt.plot(epochs, history['val_miou'], 'ro-', label='Validation MIoU')
    plt.title(f'Fold {fold+1} - Training and Validation MIoU')
    plt.xlabel('Epochs')
    plt.ylabel('MIoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fold_{fold+1}_training_curves.png'))
    plt.close()
    print(f"Saved training curves for fold {fold+1} to {save_dir}")

def visualize_predictions(images, true_masks, pred_masks, num_samples, fold, epoch, save_dir, num_classes):
    """Visualizes a few samples of images, ground truth masks, and predicted masks."""
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, images.shape[0])

    images = images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    pred_masks = pred_masks.cpu().detach()

    if num_classes == 1: # Binary
        pred_masks = (torch.sigmoid(pred_masks) > 0.5).cpu().numpy().squeeze(1) # (N, H, W)
        true_masks = true_masks.squeeze(1) # (N, H, W)
    else: # Multi-class
        pred_masks = torch.argmax(pred_masks, dim=1).cpu().numpy() # (N, H, W)
        # true_masks are already (N, H, W) with class indices

    # Denormalize images if normalization was applied
    # Assuming standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        img = images[i].transpose(1, 2, 0) # C, H, W -> H, W, C
        img = std * img + mean # Denormalize
        img = np.clip(img, 0, 1)

        tm = true_masks[i]
        pm = pred_masks[i]

        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title(f"Image {i+1}")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(tm, cmap='gray' if num_classes <= 2 else 'viridis', vmin=0, vmax=max(1, num_classes-1))
        plt.title(f"True Mask {i+1}")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pm, cmap='gray' if num_classes <= 2 else 'viridis', vmin=0, vmax=max(1, num_classes-1))
        plt.title(f"Predicted Mask {i+1}")
        plt.axis('off')

    plt.tight_layout()
    filename = f"fold_{fold+1}_epoch_{epoch}_predictions.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Saved prediction visualizations to {os.path.join(save_dir, filename)}")

if __name__ == '__main__':
    # Example Usage (binary)
    print("Testing MIoU for binary case:")
    N, C, H, W = 2, 1, 3, 3 # Batch, Channels, Height, Width
    # Mock predictions (logits)
    preds_binary = torch.randn(N, C, H, W)
    # Mock targets (0 or 1)
    targets_binary = torch.randint(0, 2, (N, C, H, W)).float()
    miou_binary = calculate_miou_batch(preds_binary, targets_binary, num_classes=1)
    print(f"Binary MIoU: {miou_binary:.4f}")

    # Example Usage (multi-class)
    print("\nTesting MIoU for multi-class case:")
    N, NUM_CLS, H, W = 2, 3, 3, 3 # Batch, Num Classes, Height, Width
    # Mock predictions (logits)
    preds_multiclass = torch.randn(N, NUM_CLS, H, W)
    # Mock targets (class indices 0, 1, 2)
    targets_multiclass = torch.randint(0, NUM_CLS, (N, H, W)) # Note: No channel dim for targets here
    miou_multiclass = calculate_miou_batch(preds_multiclass, targets_multiclass, num_classes=NUM_CLS)
    print(f"Multi-class MIoU: {miou_multiclass:.4f}")

    # Test visualization (mock data)
    print("\nTesting visualization (mock data):")
    mock_images = torch.rand(config.VISUALIZATION_COUNT, 3, 64, 64)
    mock_true_masks_binary = torch.randint(0, 2, (config.VISUALIZATION_COUNT, 1, 64, 64)).float()
    mock_pred_logits_binary = torch.randn(config.VISUALIZATION_COUNT, 1, 64, 64)

    visualize_predictions(mock_images, mock_true_masks_binary, mock_pred_logits_binary,
                          num_samples=config.VISUALIZATION_COUNT, fold=0, epoch=1,
                          save_dir=os.path.join(config.RESULTS_SAVE_PATH, 'visualizations_binary_test'),
                          num_classes=1)

    mock_true_masks_multi = torch.randint(0, NUM_CLS, (config.VISUALIZATION_COUNT, 64, 64))
    mock_pred_logits_multi = torch.randn(config.VISUALIZATION_COUNT, NUM_CLS, 64, 64)
    visualize_predictions(mock_images, mock_true_masks_multi, mock_pred_logits_multi,
                          num_samples=config.VISUALIZATION_COUNT, fold=0, epoch=1,
                          save_dir=os.path.join(config.RESULTS_SAVE_PATH, 'visualizations_multi_test'),
                          num_classes=NUM_CLS)