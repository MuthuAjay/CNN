from flask import Flask, render_template, request, send_from_directory
import os, io
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Store the original upload image
app.config['FEATURE_MAPS_FOLDER'] = 'feature_maps' # Store the feature maps
app.static_folder = 'static'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            model_name = request.form.get('model')
            feature_maps = process_image(filename, model_name)
            return render_template('index.html', filename=file.filename, model=model_name, feature_maps=feature_maps)
    return render_template('index.html', filename=None, model=None, feature_maps=[])

def process_image(image_path, model_name):
    if model_name == 'ResNet':
        model = models.resnet18(pretrained=True)
        layers = ['conv1', 'layer1_0_conv1', 'layer1_0_conv2']
        hook_layer = [model.conv1, model.layer1[0].conv1, model.layer1[0].conv2]

    elif model_name == 'VGG':
        model = models.vgg11(pretrained=True)
        layers = ['features_0','features_3','features_6']
        hook_layer = [model.features[0],model.features[3],model.features[6]]
    else:
        return {}

    model.eval()

    # Preprocess the image
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)

    # Capture layer outputs
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for i, layer in enumerate(hook_layer):
        layer.register_forward_hook(get_activation(layers[i]))

    output = model(img)

    feature_maps = {}
    for name, act in activation.items():
        num_feature_maps = act.shape[1]
        cols = min(5, num_feature_maps)
        
        fig, axes = plt.subplots(1, cols, figsize=(cols * 3, 3))
        if cols == 1:
            axes = [axes]
        
        for i in range(cols):
            axes[i].imshow(act[0, i].cpu(), cmap='viridis')
            axes[i].axis('off')

        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        # Encode the plot to base64 for embedding in HTML
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        feature_maps[name] = image_base64

    return feature_maps


@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/feature_maps/<filename>')
def send_feature_maps(filename=''):
    return send_from_directory(app.config["FEATURE_MAPS_FOLDER"], filename)

if __name__ == '__main__':
    app.run(debug=True)

