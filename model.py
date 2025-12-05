# --- model.py ---
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, FREEZE_BASE_LAYERS

class ResNet18MultiLabel(nn.Module):
    """
    Multi-label classification based on a pre-trained ResNet-50.
    The name is kept as ResNet18MultiLabel for consistency with earlier code, 
    but the architecture used is ResNet-50.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, freeze_base=FREEZE_BASE_LAYERS):
        super(ResNet18MultiLabel, self).__init__()
        
        # Load the Pre-trained Model (ResNet-50)
        self.model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Freeze all parameters if requested
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
            
        # Modify the Final Classification Layer (FC layer)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # NOTE: If we are not freezing, we must unfreeze at least the final block
        if not freeze_base:
            print("WARNING: All layers are UNFROZEN for fine-tuning.")

    def forward(self, x):
        return self.model(x)

class ViTMultiLabel(nn.Module):
    """
    Multi-label classification based on a pre-trained Vision Transformer (ViT-Base).
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, freeze_base=FREEZE_BASE_LAYERS):
        super(ViTMultiLabel, self).__init__()
        
        # Load pre-trained ViT-Base (16x16 patch size)
        self.model = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Freeze all parameters if requested
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
            
        # Replace the classifier head
        # ViT's classifier is under .heads.head
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_ftrs, num_classes)
        
        if not freeze_base:
            print("WARNING: All layers are UNFROZEN for fine-tuning.")

    def forward(self, x):
        return self.model(x)

# Example usage check:
if __name__ == '__main__':
    # Test ResNet
    resnet_model = ResNet18MultiLabel(freeze_base=True)
    print("ResNet-50 Head:", resnet_model.model.fc)
    print("ResNet FC layer requires grad:", resnet_model.model.fc.weight.requires_grad)

    # Test ViT
    vit_model = ViTMultiLabel(freeze_base=True)
    print("ViT Head:", vit_model.model.heads.head)
    print("ViT FC layer requires grad:", vit_model.model.heads.head.weight.requires_grad)