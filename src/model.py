import torch
import torch.nn as nn
import torchvision.models as models

# --- Basic U-Net Implementation (Example) ---
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # Encoder
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv_bottleneck = double_conv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up4 = double_conv(1024, 512) # 512 (from upconv) + 512 (from encoder skip)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up3 = double_conv(512, 256) # 256 (from upconv) + 256 (from encoder skip)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = double_conv(256, 128) # 128 (from upconv) + 128 (from encoder skip)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = double_conv(128, 64)   # 64 (from upconv) + 64 (from encoder skip)

        # Output layer
        self.conv_out = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv_down1(x) # -> 64
        p1 = self.pool(x1)

        x2 = self.conv_down2(p1) # -> 128
        p2 = self.pool(x2)

        x3 = self.conv_down3(p2) # -> 256
        p3 = self.pool(x3)

        x4 = self.conv_down4(p3) # -> 512
        p4 = self.pool(x4)

        # Bottleneck
        b = self.conv_bottleneck(p4) # -> 1024

        # Decoder
        u4 = self.upconv4(b) # -> 512
        # Skip connection: concatenate u4 with x4 (encoder output)
        merge4 = torch.cat([u4, x4], dim=1) # 512 + 512 = 1024
        c4 = self.conv_up4(merge4) # -> 512

        u3 = self.upconv3(c4) # -> 256
        merge3 = torch.cat([u3, x3], dim=1) # 256 + 256 = 512
        c3 = self.conv_up3(merge3) # -> 256

        u2 = self.upconv2(c3) # -> 128
        merge2 = torch.cat([u2, x2], dim=1) # 128 + 128 = 256
        c2 = self.conv_up2(merge2) # -> 128

        u1 = self.upconv1(c2) # -> 64
        merge1 = torch.cat([u1, x1], dim=1) # 64 + 64 = 128
        c1 = self.conv_up1(merge1) # -> 64

        # Output
        out = self.conv_out(c1)

        # If n_class is 1 (binary segmentation with BCEWithLogitsLoss), no activation here.
        # If n_class > 1 (multi-class with CrossEntropyLoss), no activation here as it's included in the loss.
        # If you need probabilities (e.g., for visualization or specific metrics), apply Sigmoid or Softmax later.
        return out

# --- SOTA Model Example (using segmentation-models-pytorch) ---
# You need to install this library: pip install segmentation-models-pytorch
# try:
#     import segmentation_models_pytorch as smp
# except ImportError:
#     smp = None
#     print("segmentation_models_pytorch not found. Install it to use SOTA models like Unet++, DeepLabV3+, etc.")

# def get_sota_model(model_name='Unet', encoder_name='resnet34', encoder_weights='imagenet', num_classes=1, activation=None):
#     """
#     Loads a SOTA model from segmentation_models_pytorch.
#     Args:
#         model_name (str): Name of the model (e.g., 'Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus', 'FPN', 'PSPNet', 'PAN').
#         encoder_name (str): Name of the encoder (e.g., 'resnet34', 'efficientnet-b0').
#         encoder_weights (str): Pretrained weights ('imagenet' or None).
#         num_classes (int): Number of output classes.
#         activation (str or None): Activation function for the output layer (e.g., 'sigmoid', 'softmax', None).
#                                  For BCEWithLogitsLoss, use None.
#                                  For CrossEntropyLoss, use None (activation is part of the loss).
#     """
#     if smp is None:
#         raise ImportError("segmentation_models_pytorch is not installed. Cannot load SOTA model.")

#     # Dynamically get the model class from smp
#     model_class = getattr(smp, model_name)

#     model = model_class(
#         encoder_name=encoder_name,
#         encoder_weights=encoder_weights,
#         in_channels=3, # Assuming RGB images
#         classes=num_classes,
#         activation=activation
#     )
#     return model

if __name__ == '__main__':
    # Test basic U-Net
    print("Testing Basic U-Net:")
    dummy_input = torch.randn(2, 3, 256, 256) # Batch_size=2, Channels=3, H=256, W=256
    unet_model = UNet(n_class=2) # Example: 2 classes (background + foreground)
    output = unet_model(dummy_input)
    print("Output shape:", output.shape) # Expected: (2, num_classes, 256, 256)

    # # Test SOTA model (if library is installed)
    # if smp:
    #     print("\nTesting SOTA U-Net++ from segmentation_models_pytorch:")
    #     try:
    #         sota_model = get_sota_model(
    #             model_name='UnetPlusPlus',
    #             encoder_name='resnet34',
    #             encoder_weights='imagenet',
    #             num_classes=1, # For binary segmentation with BCEWithLogitsLoss
    #             activation=None
    #         )
    #         output_sota = sota_model(dummy_input)
    #         print("SOTA Model Output shape:", output_sota.shape) # Expected: (2, 1, 256, 256)

    #         sota_multiclass_model = get_sota_model(
    #             model_name='DeepLabV3Plus',
    #             encoder_name='efficientnet-b0',
    #             encoder_weights='imagenet',
    #             num_classes=5, # Example: 5 classes
    #             activation=None # For CrossEntropyLoss
    #         )
    #         output_sota_mc = sota_multiclass_model(dummy_input)
    #         print("SOTA Multi-class Model Output shape:", output_sota_mc.shape) # Expected: (2, 5, 256, 256)

    #     except Exception as e:
    #         print(f"Error testing SOTA model: {e}")
    # else:
    #     print("\nSkipping SOTA model test as segmentation_models_pytorch is not available.")
