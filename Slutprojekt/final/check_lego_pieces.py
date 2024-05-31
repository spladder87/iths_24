import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import random
import os

# Define the same CNN architecture used for training
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

# Load the class names (adjust the path as necessary)
data_dir = "D:/Ai/dataset/tagged_images_with_lego_bricks/05 - dataset/photos/combined"



# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Load the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
# Get the class names
class_names = dataset.classes
# Load the trained model
model_path = 'best_model.pth'
num_classes = len(class_names)
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load(model_path))
model = model.cuda()  # Move the model to GPU
model.eval()  # Set the model to evaluation mode

# Function to predict the class of a LEGO piece
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.cuda()  # Move the image to GPU

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_name = class_names[class_idx]

    return class_name

# Function to load random images and check accuracy
def check_accuracy(dataset_dir, num_images=50):
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    class_names = dataset.classes

    # Randomly select images
    indices = random.sample(range(len(dataset)), num_images)
    correct_count = 0

    for idx in indices:
        image_path, true_label_idx = dataset.samples[idx]
        true_label = class_names[true_label_idx]
        predicted_label = predict(image_path)
        
        if predicted_label == true_label:
            correct_count += 1

        print(f"Image: {os.path.basename(image_path)}, True Label: {true_label}, Predicted Label: {predicted_label}")

    accuracy = correct_count / num_images
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Example usage
if __name__ == "__main__":
    check_accuracy(data_dir)
