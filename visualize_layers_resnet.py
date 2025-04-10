import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# Let's say you have a model like this
model = models.resnet18(pretrained=True)
model.eval()

# Get a sample input image
from torchvision import transforms
from PIL import Image

# Load and preprocess the image
img = Image.open(r"C:\Users\CD138JR\Downloads\profile.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img = transform(img).unsqueeze(0)  # Add batch dimension

# Define a hook to capture outputs
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Register hooks for layers you want to visualize
model.layer1[0].conv1.register_forward_hook(get_activation('conv1'))
model.layer1[0].conv2.register_forward_hook(get_activation('conv2'))

# Forward pass
output = model(img)

# Now plot the activations
for name, act in activation.items():
    num_feature_maps = act.shape[1]
    fig, axes = plt.subplots(1, min(5, num_feature_maps), figsize=(15, 5))
    fig.suptitle(f'Activations after {name}')
    for i in range(min(5, num_feature_maps)):  # Show 5 feature maps
        axes[i].imshow(act[0, i].cpu(), cmap='viridis')
        axes[i].axis('off')
    plt.show()
