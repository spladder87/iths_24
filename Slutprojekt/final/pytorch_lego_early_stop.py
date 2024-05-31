import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import logging

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('training_log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# GPU check
if not torch.cuda.is_available():
    logger.error("CUDA is not available. Please check your GPU settings.")
    sys.exit(1)
else:
    logger.info("CUDA is available.")
    logger.info("Using GPU: %s", torch.cuda.get_device_name(0))

# Define the data directory
data_dir = Path("D:/Ai/dataset/tagged_images_with_lego_bricks/05 - dataset/photos/combined")

# Define transformations for training and validation
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Get the class names
class_names = dataset.classes
logger.info("Classes: %s", class_names)

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 22 * 22, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate the model
num_classes = len(class_names)
model = SimpleCNN(num_classes)
model = model.cuda()  # Move the model to GPU

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
early_stopping_patience = 5
best_val_loss = float('inf')
patience_counter = 0

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    global best_val_loss, patience_counter
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        logger.info('Epoch %d/%d, Loss: %.4f', epoch + 1, epochs, epoch_loss)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
            logger.info('Validation loss decreased. Saving the model.')
        else:
            patience_counter += 1
            logger.info('Validation loss did not decrease for %d epochs.', patience_counter)
            if patience_counter >= early_stopping_patience:
                logger.info('Early stopping triggered. Stopping training.')
                return train_losses, val_losses, val_accuracies
    
    return train_losses, val_losses, val_accuracies

# Validation function
def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total
    logger.info('Validation Loss: %.4f, Accuracy: %.4f', epoch_loss, accuracy)
    return epoch_loss, accuracy

# Train the model
epochs = 50
train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

# Plot the training and validation losses and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.savefig('training_validation_plot.png')
plt.show()

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model = model.cuda()

# Convert the model to TorchScript
model.eval()
example_input = torch.randn(1, 3, 180, 180).cuda()
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

logger.info("Model training and conversion complete.")
