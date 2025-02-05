# 31-01-2025

# 
import requests
import torch
from torchvision import transforms
from PIL import Image
import os

def predict_image(image_path, model_path):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64*15*15, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

if __name__ == "__main__":
    image_path = 'data/test/sample.jpg'
    model_path = 'model/model.pth'
    prediction = predict_image(image_path, model_path)
    print(f'Predicted class: {prediction}')





