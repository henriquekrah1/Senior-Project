import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim

# Paths dataset
train_dir = r"C:\Users\Henrique\Desktop\Senior Project\dataset\train"
val_dir = r"C:\Users\Henrique\Desktop\Senior Project\dataset\val"

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to EfficientNet's input size
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load EfficientNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EfficientNet.from_pretrained('efficientnet-b0')
num_features = model._fc.in_features
model._fc = nn.Linear(num_features, 2)  # Binary classification
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10
for epoch in range(epochs):
    model.train()  
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    
    model.eval()  
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print epoch results
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "efficientnet_finetuned.pth")
print("Model saved as efficientnet_finetuned.pth")
