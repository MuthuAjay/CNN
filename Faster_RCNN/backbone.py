import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional, Dict, Any

# # import vgg16

# def vgg16(pretrained=True):
    
#     """Load the VGG16 model with pretrained weights."""
#     model = models.vgg16(pretrained=pretrained)
    
#     print(model)
    
# if __name__ == "__main__":
#     vgg16(pretrained=True)
        
        
# class VGGBlock(nn.Module):
    
#     def __init__(self, in_channels: int, out_channels:int, num_convs:int):
#         super(VGGBlock, self).__init__()
        
#         layers = []
        
#         for i in range(num_convs):
#             layers.extend([
#                 nn.Conv2d(in_channels if i==0 else out_channels,
#                           out_channels,
#                           kernel_size=3,
#                           stride=1,
#                           padding=1),
#                 nn.ReLU(inplace=True)
#             ])
#         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         self.block = nn.Sequential(*layers)
        
#     def forward(self, x):
#         return self.block(x)
    
    
class VGG(nn.Module):
    
    VGG_CONFIGS = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    }
        
    def __init__(self,
                architecture: str = 'VGG16',
                num_classes: int = 1000,
                input_channels: int = 3,
                dropout_rate: float = 0.5,
                use_batch_norm: bool = False,
                pretrained: bool = False,
                ):
        
        """
        Args:
            architecture (str): The VGG architecture to use (e.g., 'VGG16').
            num_classes (int): Number of output classes.
            input_channels (int): Number of input channels (default is 3 for RGB).
            dropout_rate (float): Dropout rate for the classifier.
            use_batch_norm (bool): Whether to use batch normalization.
            pretrained (bool): If True, load pretrained weights.
        """
        
        super(VGG, self).__init__()
        
        if architecture not in self.VGG_CONFIGS:
            raise ValueError(f"Unsupported architecture: {architecture}. Supported architectures: {list(self.VGG_CONFIGS.keys())}")
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build the feauture extractor
        self.features = self._make_layers(
            config=self.VGG_CONFIGS[architecture],
            input_channels=input_channels,
            use_batch_norm=use_batch_norm
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Adaptive pooling to ensure output size is fixed
        
        # Build the classifier
        self.classifier = self._make_classifier(num_classes=num_classes, dropout_rate=dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
        
        if pretrained:
            self._load_pretrained_weights(architecture)
        
        
    def _make_layers(self, config: List, input_channels: int, use_batch_norm: bool = False) -> nn.Sequential:
        
        layers = []
        in_channels = input_channels
        
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=1)
                if use_batch_norm:
                    layers.extend([
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ])
                else:
                    layers.extend([
                        conv2d,
                        nn.ReLU(inplace=True)
                    ])
                in_channels = v
                
        return nn.Sequential(*layers)
        
    def _make_classifier(self, num_classes: int, dropout_rate: float) -> nn.Sequential:
        """Build the classifier part of the VGG model."""
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Assuming input size is 224x224
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
        
    def _initialize_weights(self):
        """Initialize weights for the model."""
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def _load_pretrained_weights(self):
        """Load pretrained weights for the model."""
        
        try:
            import torchvision.models as models
            pretrained_dict = getattr(models, f'vgg{self.architecture[-2:]}')(pretrained=True).state_dict()
            model_dict = self.state_dict()
            
            # Filter out unnecessary keys and size mismatches
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.size() == model_dict[k].size()}
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded pretrained {self.architecture} weights")
            
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x: torch.Tensor, layer_idx: Optional[int] =None) -> torch.Tensor:
        """Get feature maps from a specific layer."""
        if layer_idx is None:
            return self.features(x)
        
        for i , layer in enumerate(self.features):
            x = layer(x)
            if i == layer_idx:
                return x
        raise IndexError(f"Layer index {layer_idx} out of range for feature extractor with {len(self.features)} layers.")
    
    def freeze_layers(self):
        """Freeze all layers except the classifier."""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self):
        """Unfreeze all layers."""
        for param in self.features.parameters():
            param.requires_grad = True
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': self.architecture,
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32 weights
        }
        
        
# Factory function for easy model creation
def create_vgg(architecture: str = 'VGG16', **kwargs) -> VGG:
    """Factory function to create VGG models"""
    return VGG(architecture=architecture, **kwargs)

# Example usage
if __name__ == "__main__":
    # Create different VGG variants
    models = {
        'VGG16': create_vgg('VGG16', num_classes=10),
        'VGG19': create_vgg('VGG19', num_classes=1000, use_batch_norm=True),
        'VGG11': create_vgg('VGG11', num_classes=100, dropout_rate=0.3)
    }
    
    # Test each model
    for name, model in models.items():
        print(f"\n{name} Model Info:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Test feature extraction
        features = model.get_feature_maps(x)
        print(f"  Feature maps shape: {features.shape}")
