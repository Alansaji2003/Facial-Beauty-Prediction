from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask_cors import CORS
import os

# Define the model architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, bias=False)
        self.relu_pool1 = bn_relu_pool(inplanes=96)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, padding=2, groups=2, bias=False)
        self.relu_pool2 = bn_relu_pool(inplanes=192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu3 = bn_relu(inplanes=384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu4 = bn_relu(inplanes=384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu_pool5 = bn_relu_pool(inplanes=256)
        # classifier
        self.conv6 = nn.Conv2d(256, 256, kernel_size=5, groups=2, bias=False)
        self.relu6 = bn_relu(inplanes=256)
        self.conv7 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        
    def forward(self, x):
        # Define the forward pass of your model
        x = self.conv1(x)
        x = self.relu_pool1(x)
        x = self.conv2(x)
        x = self.relu_pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu_pool5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        return x

def bn_relu(inplanes):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True))

def bn_relu_pool(inplanes, kernel_size=3, stride=2):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

# Load the trained model
def load_model(model_path, model):
    print("Loading model...")
    pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'), encoding='latin1')
    model_dict = model.state_dict()
    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Model loaded successfully!")

# Preprocess the input image
def preprocess_image(image):
    print("Preprocessing image...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# Initialize Flask application
app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')
# Define a route to receive image data and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    print("Received POST request...")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        try:
            # Load the model
            model = AlexNet().cuda()
            load_model('trained_models_for_pytorch/models/alexnet.pth', model)
            model.eval()  # Set the model to evaluation mode
            # Read and preprocess the image
            image = Image.open(file).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0).cuda()
            # Make predictions
            with torch.no_grad():
                print("Making predictions...")
                output = model(image_tensor)
                print("Predictions made successfully!")
            # Process the output
            # Encode prediction result as JSON using UTF-8 encoding
            prediction = output.item()
            return jsonify({'prediction': prediction})
        except Exception as e:
            error_message = str(e).encode('utf-8')  # Encode error message as UTF-8
            return jsonify({'error': error_message.decode('utf-8')})  # Decode error message for JSON serialization


if __name__ == '__main__':
    app.run(debug=True)
