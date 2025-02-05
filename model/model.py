# model/model.py
import torch.nn as nn
from torchvision.models import resnet18

def create_model():
    """
    Cria e retorna uma instância do modelo ResNet18 modificada para tons de cinza.
    A camada de entrada é ajustada para 1 canal (tons de cinza), e a camada de saída
    é ajustada para 2 classes (osteoporose e normal).
    """
    model = resnet18(pretrained=False)  # Não carregar pesos pré-treinados
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Ajuste para 1 canal de entrada
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: osteoporose e normal
    return model