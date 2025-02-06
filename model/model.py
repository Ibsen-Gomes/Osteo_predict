import torch
import torch.nn as nn
from torchvision import models

class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        # ✅ 1. Melhorando a primeira convolução para capturar mais padrões
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)  # Agora com 64 filtros
        self.model.bn1 = nn.BatchNorm2d(32)  # Adicionando BatchNorm
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remover a camada totalmente conectada original

        # ✅ 2. Melhorando a extração de características (camadas convolucionais adicionais)
        self.additional_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Determinar o tamanho da entrada da camada Linear automaticamente
        self._calculate_fc_input_size()

        # ✅ 3. Melhorando a camada totalmente conectada
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 2 classes: osteoartrite vs normal
        )

    def _calculate_fc_input_size(self):
        """ Calcula dinamicamente o tamanho da camada totalmente conectada """
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224)  # Simula uma imagem de entrada
            x = self.model.conv1(dummy_input)
            x = self.model.bn1(x)  # Aplicando a normalização de batch
            x = self.additional_conv(x)
            self.fc_input_size = x.view(1, -1).size(1)  # Calcula a saída antes da camada totalmente conectada

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)  # Aplicando a normalização de batch
        x = self.additional_conv(x)
        
        x = x.view(x.size(0), -1)  # Achatar para entrada da FC
        x = self.fc(x)
        return x

def create_model():
    model = ModifiedResNet18()
    return model
