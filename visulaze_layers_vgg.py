import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Load pretrained VGG16 model
model = models.vgg16(pretrained=True)
model.eval()

# Preprocess your input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_path = r"C:\Users\CD138JR\Downloads\transformer_gpt.jpg"
# Load an image
# img = Image.open(r"C:\Users\CD138JR\Downloads\profile.jpg")  # <-- Replace with your image path
img = Image.open(image_path)
img = transform(img).unsqueeze(0)    # Add batch dimension

# Hook to capture activations
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Register hooks to all Conv layers
conv_layers = []
model_children = list(model.features.children())
for i, layer in enumerate(model_children):
    if isinstance(layer, nn.Conv2d):
        conv_layers.append(layer)

# Attach hook to each conv layer
for idx, layer in enumerate(conv_layers):
    layer.register_forward_hook(get_activation(f'conv_{idx}'))

# Forward pass
output = model(img)

# # Plot the feature maps
# for name, feature_map in activation.items():
#     print(f"Visualizing {name} with shape {feature_map.shape}")
#     num_feature_maps = feature_map.shape[1]  # Channels
    
#     # Plot only a few feature maps for visualization
#     fig, axes = plt.subplots(1, min(5, num_feature_maps), figsize=(20, 5))
#     fig.suptitle(f'Feature maps after {name}', fontsize=16)
    
#     for i in range(min(5, num_feature_maps)):  # Plot 5 feature maps
#         axes[i].imshow(feature_map[0, i].cpu(), cmap='viridis')
#         axes[i].axis('off')
    
#     plt.show()

# plot in a single image
fig, axes = plt.subplots(len(activation), 5, figsize=(20, 5 * len(activation)))
for idx, (name, feature_map) in enumerate(activation.items()):
    num_feature_maps = feature_map.shape[1]  # Channels
    fig.suptitle(f'Feature maps after {name}', fontsize=16)
    for i in range(min(5, num_feature_maps)):  # Plot 5 feature maps
        axes[idx, i].imshow(feature_map[0, i].cpu(), cmap='viridis')
        axes[idx, i].axis('off')
plt.show()