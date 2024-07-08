import io
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from captcha_cnn import CaptchaCNN


app = Flask(__name__)


transform = transforms.Compose([
    transforms.Resize((38, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

model = CaptchaCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cpu")
model.load_state_dict(torch.load("model/captcha_model_1000_ry.pth", map_location=torch.device('cpu')))
model.to(device)  # 将模型移到GPU
model.eval()


def predict_image(file):
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output, 2)
    return ''.join(str(digit.item()) for digit in predicted[0])


@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        predict_result = predict_image(file)
        json_result = {
            "predict": str(predict_result)
        }
        return jsonify(json_result), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=15556, debug=False)
