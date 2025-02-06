import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import sys
import time  # 📌 Biblioteca para medir o tempo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import create_model  # ✅ Importa a arquitetura da CNN

# 🔹 Definir transformações para imagens (tons de cinza, resize e normalização)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 🔹 Carregar dataset
dataset = datasets.ImageFolder(root='data', transform=transform)

# 🔹 Dividir em treino (80%) e teste (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 🔹 Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 🔹 Criar o modelo
model = create_model()

# 🔹 Definir função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔹 Enviar modelo para GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔹 Iniciar contagem do tempo
start_time = time.time()
print("\n🚀 Iniciando treinamento...")
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 🔹 Avaliar no conjunto de teste
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Obtém a classe predita
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, 🎯 Precisão: {accuracy:.2f}%")

# 🔹 Finalizar contagem do tempo
end_time = time.time()
execution_time = end_time - start_time
print(f"\n⏳ Tempo total de treinamento: {execution_time:.2f} segundos")

# 🔹 Salvar o modelo treinado
torch.save(model.state_dict(), 'model/model.pth')
print("\n✅ Modelo treinado e salvo em model/model.pth")
