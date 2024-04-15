import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


# Load your data
X = np.load('X3.npy')
Y = np.load('Y3.npy')
X=X[:1000]
Y=Y[:1000]
for i in range(len(Y)):
  Y[i]=Y[i]-1
# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define a custom dataset
class YogaPoseDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

# Define a simple convolutional neural network model
class YogaPoseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(YogaPoseClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Create data loaders
train_dataset = YogaPoseDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = YogaPoseDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=64)

# Initialize the model
model = YogaPoseClassifier(num_classes=11)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
accuracy_data=[]
epoch_data=[]
loss_data = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0 
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x.unsqueeze(1).float())
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.update(1)
        loss_data.append(epoch_loss / len(train_loader))

    # Evaluate the model on the validation set
    model.eval()

    with torch.no_grad():
        total = 0
        correct = 0
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x.unsqueeze(1).float())
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        accuracy = 100 * correct / total
        accuracy_data.append(accuracy)
        epoch_data.append(epoch + 1)
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Validation Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), 'yoga_pose_model.pth')
