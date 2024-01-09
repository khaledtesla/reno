from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)


# Load the model
loaded_model = models.densenet121()
num_features = loaded_model.classifier.in_features
loaded_model.classifier = nn.Linear(num_features, 5)
loaded_model.load_state_dict(torch.load('derma_diseases_detection_best.pt', map_location=torch.device('cpu')))

loaded_model.eval()

# Define the image preprocessing function
def preprocess_image(image):
    image = Image.open(io.BytesIO(image))
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Define the prediction function
def predict_skin_disease(image):
    preprocessed_image = preprocess_image(image)
    with torch.no_grad():
        output = loaded_model(preprocessed_image)
        _, predicted_class = torch.max(output, 1)
    class_label = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    class_label = class_label[predicted_class.item()]
    return class_label

# Define Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_bytes = file.read()
        result = predict_skin_disease(image_bytes)
        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, port=5001)