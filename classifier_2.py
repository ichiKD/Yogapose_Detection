import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load your data
X = np.load('X3.npy')
Y = np.load('Y3.npy')
X = X[:2000]
Y = Y[:2000]

for i in range(len(Y)):
    Y[i] = Y[i] - 1

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

class WaveletLinearLayer(nn.Module):
    def __init__(self, input_size, output_size, wavelet_name='haar'):
        super(WaveletLinearLayer, self).__init__()
        self.wavelet_name = wavelet_name
        transformed_size = 4 * input_size * input_size 
        self.linear = nn.Linear(transformed_size, output_size)
    def wavelet_transform(self, x):
        # Reshape the tensor to 2D before applying wavelet transform
        size = int(np.sqrt(x.shape[0] // 4))  # Dividing by 4 for the channels
        x_2d = x.view(4, size, size).detach().cpu().numpy()
        # Apply wavelet transform on each channel
        transformed_channels = []
        for i in range(4):
            coeffs2 = pywt.dwt2(x_2d[i], self.wavelet_name)
            LL, (LH, HL, HH) = coeffs2
            transformed_x = np.concatenate([LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()])
            transformed_channels.append(transformed_x)
        return np.concatenate(transformed_channels) # Concatenate transformed channels
    def forward(self, x):
        with torch.no_grad():
            batch_transformed = [self.wavelet_transform(image) for image in x]
            batch_transformed_tensor = torch.tensor(batch_transformed, dtype=torch.float).to(x.device)
        return self.linear(batch_transformed_tensor) # Passing through the linear layer

# Define a custom dataset
class YogaPoseDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        # Apply wavelet transform
        coeffs2 = pywt.dwt2(x, 'haar')
        LL, (LH, HL, HH) = coeffs2
        x_transformed = np.stack([LL, LH, HL, HH], axis=0)
        return torch.tensor(x_transformed, dtype=torch.float), y

# Define a simple convolutional neural network model
class YogaPoseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(YogaPoseClassifier, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Adjust the input size of the first linear layer based on the output size of the conv layers
        self.fc1 = WaveletLinearLayer(64, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc1(x) # Apply the wavelet transform and linear layer
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

for epoch in range(num_epochs):
    model.train()
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x.float())
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            pbar.update(1)

    # Evaluate the model on the validation set
    model.eval()

    with torch.no_grad():
        total = 0
        correct = 0
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x.float())
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Validation Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), 'yoga_pose_model.pth')
