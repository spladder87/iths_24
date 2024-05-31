Here's a detailed Markdown report for your script, formatted for submission to your class:

---

# Report: Image Classification using PyTorch with GPU Acceleration and Early Stopping

## Introduction

This report outlines the process of training an image classification model using PyTorch with GPU acceleration and early stopping. The dataset used consists of images of LEGO bricks, divided into different categories. The objective is to build a Convolutional Neural Network (CNN) that can accurately classify these images. The script includes steps for data loading, preprocessing, model definition, training, validation, early stopping, saving, and converting the model for deployment using TorchScript.

## Environment Setup

Before starting with the implementation, ensure that PyTorch with CUDA support is installed. This can be achieved by running the following command:

```bash
pip install torch torchvision matplotlib numpy tqdm
```

## GPU Check

The script begins by checking if a GPU is available. If CUDA is not available, the script exits with an error message.

```python
if not torch.cuda.is_available():
    logger.error("CUDA is not available. Please check your GPU settings.")
    sys.exit(1)
else:
    logger.info("CUDA is available.")
    logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
```

## Data Loading and Preprocessing

### Data Directory

The dataset is stored in a directory with subdirectories for each class. The images are pre-labeled based on the directory they are stored in.

```python
data_dir = Path("D:/Ai/dataset/tagged_images_with_lego_bricks/05 - dataset/photos/combined")
```

### Transformations

To prepare the data for training, we apply transformations such as resizing, normalization, and conversion to tensors.

```python
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Dataset and DataLoader

We load the dataset using `datasets.ImageFolder` and split it into training and validation sets. DataLoader is used to handle batching and shuffling.

```python
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

## Model Definition

### CNN Architecture

We define a simple CNN with three convolutional layers, max pooling, dropout, and two fully connected layers.

```python
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
```

### Instantiate the Model

We instantiate the model and move it to the GPU.

```python
num_classes = len(dataset.classes)
model = SimpleCNN(num_classes)
model = model.cuda()  # Move the model to GPU
```

## Training and Validation

### Loss Function and Optimizer

We use CrossEntropyLoss and Adam optimizer.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Early Stopping Parameters

Early stopping is implemented to prevent overfitting by stopping the training process if the validation loss does not improve after a certain number of epochs.

```python
early_stopping_patience = 5
best_val_loss = float('inf')
patience_counter = 0
```

### Training Function

The training function performs forward and backward passes, updates the model parameters, and includes a progress bar using `tqdm`. It also checks for early stopping criteria.

```python
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
```

### Validation Function

The validation function evaluates the model on the validation dataset and computes accuracy.

```python
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
```

### Training the Model

We train the model for a specified number of epochs and visualize the training progress.

```python
epochs = 50
train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
```

## Plotting Results

We plot the training and validation losses and accuracy.

```python
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
```

## Saving and Loading the Model

### Save the Model

We save the best model's state dictionary.

```python
torch.save(model.state_dict(), 'best_model.pth')
```

### Load the Best Model

We load the best model for further use.

```python
model.load_state_dict(torch.load('best_model.pth'))
model = model.cuda()
```

## Model Conversion

### Convert to TorchScript

We convert the trained model to TorchScript for deployment.

```python
model.eval()
example_input = torch.randn(1, 3, 180, 180).cuda()
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")
```

## Conclusion

This report provided a step-by-step guide on how to implement an image classification model using PyTorch with GPU acceleration and early stopping. By leveraging the power of GPUs and early stopping, we can significantly speed up the training process and prevent overfitting, enabling the handling of large datasets more efficiently. The model can be further improved by experimenting with different architectures, hyperparameters, and data

 augmentation techniques.

## Training Log

Here is an excerpt from the training log, showing the GPU detection, class names, training and validation losses, and accuracies for each epoch.

```plaintext
2024-05-30 21:20:21,489 - INFO - CUDA is available.
2024-05-30 21:20:21,490 - INFO - Using GPU: NVIDIA GeForce RTX 3080 Laptop GPU
2024-05-30 21:20:23,628 - INFO - Classes: ['10197', '10201', '10288', '10314', '10928', '11090', '11153', '11211', '11212', '11213', '11214', '11215', '11272', '11458', '11476', '11477', '11478', '13349', '13547', '13548', '13731', '14395', '14417', '14419', '14704', '14716', '14720', '14769', '15068', '15070', '15092', '15100', '15207', '15254', '15332', '15379', '15395', '15397', '15460', '15461', '15470', '15535', '15573', '15672', '15706', '15712', '16577', '17114', '17485', '18649', '18651', '18653', '18674', '18677', '18838', '18969', '19159', '20482', '20896', '21229', '22385', '22388', '22390', '22391', '22484', '22885', '22888', '22889', '22890', '22961', '2357', '23969', '24014', '24122', '2412b', '2419', '2420', '24246', '24299', '2431', '24316', '24375', '2441', '2445', '2450', '24505', '2453b', '2454', '2456', '2458', '2460', '2476a', '2486', '24866', '25269', '2540', '25893', '26047', '2639', '2654', '26601', '26604', '27255', '27262', '27266', '2730', '2736', '27940', '2853', '2854', '2877', '2904', '2921', '2926', '30000', '3001', '3002', '3003', '3004', '30044', '30046', '3005', '3007', '3008', '3009', '30099', '3010', '30136', '30157', '30165', '3020', '3021', '3022', '3023', '30237b', '3024', '3028', '3031', '3032', '3033', '3034', '3035', '30357', '30361c', '30363', '30367c', '3037', '3038', '30387', '3039', '3040b', '30414', '3045', '30503', '30552', '30553', '30565', '3062', '3068', '3069b', '3070b', '3185', '32000', '32002', '32013', '32014', '32015', '32016', '32017', '32028', '32034', '32039', '32054', '32056', '32059', '32062', '32064a', '32073', '32123b', '32124', '32126', '32140', '32184', '32187', '32192', '32198', '32250', '32278', '32291', '32316', '32348', '32449', '3245', '32523', '32524', '32526', '32529', '32530', '32557', '32828', '32932', '3298', '33291', '33299b', '33909', '3460', '35044', '35336_4176', '3622', '3623', '3633', '3639', '3640', '3659', '3660', '3665', '3666', '3673', '3676', '3679', '3680', '36840', '36841', '3700', '3701', '3705', '3706', '3707', '3710', '3713', '3747b', '3749', '3795', '3823', '3832', '3895', '3941', '3942c', '3957', '3958', '3960', '39739', '4032a', '40490', '4070', '4081b', '4083', '40902', '41530', '41532', '4162', '41677', '41678', '41682', '41740', '41747', '41748', '41768', '41769', '41770', '4185', '42003', '42022', '42023', '4216', '4218', '42446', '4274', '4282', '4286', '4287b', '43708', '43712', '43713', '43719', '43898', '44126', '44568', '4460b', '44728', '4477', '44809', '44861', '4488', '4490', '4510', '4519', '45590', '45677', '4600', '4727', '4733', '47397', '47398', '4740', '4742', '47456', '47753', '47755', '47905', '48092', '48169', '48171', '48336', '4865b', '4871', '48723', '48729b', '48933', '48989', '49668', '50304', '50305', '50373', '50950', '51739', '52031', '52501', '54200', '54383', '54384', '55013', '57519', '57520', '57585', '57895', '57909b', '58090', '58176', '59426', '59443', '60032', '6005', '6014', '6020', '60470b', '60471', '60474', '60475b', '60476', '60477', '60478', '60479', '60481', '60483', '60484', '60581', '60592', '60593', '60594', '60596', '60598', '60599', '6060', '60607', '60608', '60616b', '60621', '60623', '6081', '60897', '6091', '61070', '61071', '6111', '61252', '61409', '6141', '61485', '6157', '6182', '6215', '6222', '6231', '6232', '6233', '62462', '63864', '63868', '63869', '64225', '64288', '64391', '64393', '64681', '64683', '64712', '6536', '6541', '6553', '6558', '6587', '6628', '6632', '6636', '72454', '74261', '84954', '85080', '85861', '85943', '85984', '87079', '87081', '87082', '87083', '87087', '87544', '87552', '87580', '87609', '87620', '87697', '88292', '88323', '88646', '88930', '90195', '90202', '90609', '90611', '90630', '92013', '92092', '92579', '92582', '92583', '92589', '92907', '92947', '92950', '93273', '93274', '93606', '94161', '98100', '98138', '98197', '98262', '98283', '98560', '99008', '99021', '99206', '99207', '99773', '99780', '99781']
2024-05-30 21:41:51,320 - INFO - Epoch 1/15, Loss: 4.0566
2024-05-30 21:46:47,821 - INFO - Validation Loss: 2.8269, Accuracy: 0.3395
2024-05-30 21:49:46,568 - INFO - CUDA is available.
2024-05-30 21:49:46,582 - INFO - Using GPU: NVIDIA GeForce RTX 3080 Laptop GPU
2024-05

-30 21:49:48,593 - INFO - Classes: ['10197', '10201', '10288', '10314', '10928', '11090', '11153', '11211', '11212', '11213', '11214', '11215', '11272', '11458', '11476', '11477', '11478', '13349', '13547', '13548', '13731', '14395', '14417', '14419', '14704', '14716', '14720', '14769', '15068', '15070', '15092', '15100', '15207', '15254', '15332', '15379', '15395', '15397', '15460', '15461', '15470', '15535', '15573', '15672', '15706', '15712', '16577', '17114', '17485', '18649', '18651', '18653', '18674', '18677', '18838', '18969', '19159', '20482', '20896', '21229', '22385', '22388', '22390', '22391', '22484', '22885', '22888', '22889', '22890', '22961', '2357', '23969', '24014', '24122', '2412b', '2419', '2420', '24246', '24299', '2431', '24316', '24375', '2441', '2445', '2450', '24505', '2453b', '2454', '2456', '2458', '2460', '2476a', '2486', '24866', '25269', '2540', '25893', '26047', '2639', '2654', '26601', '26604', '27255', '27262', '27266', '2730', '2736', '27940', '2853', '2854', '2877', '2904', '2921', '2926', '30000', '3001', '3002', '3003', '3004', '30044', '30046', '3005', '3007', '3008', '3009', '30099', '3010', '30136', '30157', '30165', '3020', '3021', '3022', '3023', '30237b', '3024', '3028', '3031', '3032', '3033', '3034', '3035', '30357', '30361c', '30363', '30367c', '3037', '3038', '30387', '3039', '3040b', '30414', '3045', '30503', '30552', '30553', '30565', '3062', '3068', '3069b', '3070b', '3185', '32000', '32002', '32013', '32014', '32015', '32016', '32017', '32028', '32034', '32039', '32054', '32056', '32059', '32062', '32064a', '32073', '32123b', '32124', '32126', '32140', '32184', '32187', '32192', '32198', '32250', '32278', '32291', '32316', '32348', '32449', '3245', '32523', '32524', '32526', '32529', '32530', '32557', '32828', '32932', '3298', '33291', '33299b', '33909', '3460', '35044', '35336_4176', '3622', '3623', '3633', '3639', '3640', '3659', '3660', '3665', '3666', '3673', '3676', '3679', '3680', '36840', '36841', '3700', '3701', '3705', '3706', '3707', '3710', '3713', '3747b', '3749', '3795', '3823', '3832', '3895', '3941', '3942c', '3957', '3958', '3960', '39739', '4032a', '40490', '4070', '4081b', '4083', '40902', '41530', '41532', '4162', '41677', '41678', '41682', '41740', '41747', '41748', '41768', '41769', '41770', '4185', '42003', '42022', '42023', '4216', '4218', '42446', '4274', '4282', '4286', '4287b', '43708', '43712', '43713', '43719', '43898', '44126', '44568', '4460b', '44728', '4477', '44809', '44861', '4488', '4490', '4510', '4519', '45590', '45677', '4600', '4727', '4733', '47397', '47398', '4740', '4742', '47456', '47753', '47755', '47905', '48092', '48169', '48171', '48336', '4865b', '4871', '48723', '48729b', '48933', '48989', '49668', '50304', '50305', '50373', '50950', '51739', '52031', '52501', '54200', '54383', '54384', '55013', '57519', '57520', '57585', '57895', '57909b', '58090', '58176', '59426', '59443', '60032', '6005', '6014', '6020', '60470b', '60471', '60474', '60475b', '60476', '60477', '60478', '60479', '60481', '60483', '60484', '60581', '60592', '60593', '60594', '60596', '60598', '60599', '6060', '60607', '60608', '60616b', '60621', '60623', '6081', '60897', '6091', '61070', '61071', '6111', '61252', '61409', '6141', '61485', '6157', '6182', '6215', '6222', '6231', '6232', '6233', '62462', '63864', '63868', '63869', '64225', '64288', '64391', '64393', '64681', '64683', '64712', '6536', '6541', '6553', '6558', '6587', '6628', '6632', '6636', '72454', '74261', '84954', '85080', '85861', '85943', '85984', '87079', '87081', '87082', '87083', '87087', '87544', '87552', '87580', '87609', '87620', '87697', '88292', '88323', '88646', '88930', '90195', '90202', '90609', '90611', '90630', '92013', '92092', '92579', '92582', '92583', '92589', '92907', '92947', '92950', '93273', '93274', '93606', '94161', '98100', '98138', '98197', '98262', '98283', '98560', '99008', '99021', '99206', '99207', '99773', '99780', '99781']
2024-05-30 22:05:27,708 - INFO - Epoch 1/50, Loss: 3.5815
2024-05-30 22:09:07,244 - INFO - Validation Loss: 2.2947, Accuracy: 0.4436
2024-05-30 22:09:08,169 - INFO - Validation loss decreased. Saving the model.
2024-05-30 22:23:55,083 - INFO - Epoch 2/50, Loss: 2.5078
2024-05-30 22:27:16,954 - INFO - Validation Loss: 2.0029, Accuracy: 0.5036
2024-05-30 22:27:17,899 - INFO - Validation loss decreased. Saving the model.
2024-05-30 22:42:00,932 - INFO - Epoch 3/50, Loss: 2.2039
2024-

05-30 22:45:25,978 - INFO - Validation Loss: 1.7968, Accuracy: 0.5452
2024-05-30 22:45:26,906 - INFO - Validation loss decreased. Saving the model.
2024-05-30 23:00:02,769 - INFO - Epoch 4/50, Loss: 2.0271
2024-05-30 23:03:25,599 - INFO - Validation Loss: 1.7553, Accuracy: 0.5557
2024-05-30 23:03:26,524 - INFO - Validation loss decreased. Saving the model.
2024-05-30 23:18:20,734 - INFO - Epoch 5/50, Loss: 1.9059
2024-05-30 23:21:47,370 - INFO - Validation Loss: 1.6752, Accuracy: 0.5761
2024-05-30 23:21:48,314 - INFO - Validation loss decreased. Saving the model.
2024-05-31 07:30:12,535 - INFO - CUDA is available.
2024-05-31 07:30:12,552 - INFO - Using GPU: NVIDIA GeForce RTX 3080 Laptop GPU
2024-05-31 07:30:30,010 - INFO - Classes: ['10197', '10201', '10288', '10314', '10928', '11090', '11153', '11211', '11212', '11213', '11214', '11215', '11272', '11458', '11476', '11477', '11478', '13349', '13547', '13548', '13731', '14395', '14417', '14419', '14704', '14716', '14720', '14769', '15068', '15070', '15092', '15100', '15207', '15254', '15332', '15379', '15395', '15397', '15460', '15461', '15470', '15535', '15573', '15672', '15706', '15712', '16577', '17114', '17485', '18649', '18651', '18653', '18674', '18677', '18838', '18969', '19159', '20482', '20896', '21229', '22385', '22388', '22390', '22391', '22484', '22885', '22888', '22889', '22890', '22961', '2357', '23969', '24014', '24122', '2412b', '2419', '2420', '24246', '24299', '2431', '24316', '24375', '2441', '2445', '2450', '24505', '2453b', '2454', '2456', '2458', '2460', '2476a', '2486', '24866', '25269', '2540', '25893', '26047', '2639', '2654', '26601', '26604', '27255', '27262', '27266', '2730', '2736', '27940', '2853', '2854', '2877', '2904', '2921', '2926', '30000', '3001', '3002', '3003', '3004', '30044', '30046', '3005', '3007', '3008', '3009', '30099', '3010', '30136', '30157', '30165', '3020', '3021', '3022', '3023', '30237b', '3024', '3028', '3031', '3032', '3033', '3034', '3035', '30357', '30361c', '30363', '30367c', '3037', '3038', '30387', '3039', '3040b', '30414', '3045', '30503', '30552', '30553', '30565', '3062', '3068', '3069b', '3070b', '3185', '32000', '32002', '32013', '32014', '32015', '32016', '32017', '32028', '32034', '32039', '32054', '32056', '32059', '32062', '32064a', '32073', '32123b', '32124', '32126', '32140', '32184', '32187', '32192', '32198', '32250', '32278', '32291', '32316', '32348', '32449', '3245', '32523', '32524', '32526', '32529', '32530', '32557', '32828', '32932', '3298', '33291', '33299b', '33909', '3460', '35044', '35336_4176', '3622', '3623', '3633', '3639', '3640', '3659', '3660', '3665', '3666', '3673', '3676', '3679', '3680', '36840', '36841', '3700', '3701', '3705', '3706', '3707', '3710', '3713', '3747b', '3749', '3795', '3823', '3832', '3895', '3941', '3942c', '3957', '3958', '3960', '39739', '4032a', '40490', '4070', '4081b', '4083', '40902', '41530', '41532', '4162', '41677', '41678', '41682', '41740', '41747', '41748', '41768', '41769', '41770', '4185', '42003', '42022', '42023', '4216', '4218', '42446', '4274', '4282', '4286', '4287b', '43708', '43712', '43713', '43719', '43898', '44126', '44568', '4460b', '44728', '4477', '44809', '44861', '4488', '4490', '4510', '4519', '45590', '45677', '4600', '4727', '4733', '47397', '47398', '4740', '4742', '47456', '47753', '47755', '47905', '48092', '48169', '48171', '48336', '4865b', '4871', '48723', '48729b', '48933', '48989', '49668', '50304', '50305', '50373', '50950', '51739', '52031', '52501', '54200', '54383', '54384', '55013', '57519', '57520', '57585', '57895', '57909b', '58090', '58176', '59426', '59443', '60032', '6005', '6014', '6020', '60470b', '60471', '60474', '60475b', '60476', '60477', '60478', '60479', '60481', '60483', '60484', '60581', '60592', '60593', '60594', '60596', '60598', '60599', '6060', '60607', '60608', '60616b', '60621', '60623', '6081', '60897', '6091', '61070', '61071', '6111', '61252', '61409', '6141', '61485', '6157', '6182', '6215', '6222', '6231', '6232', '6233', '62462', '63864', '63868', '63869', '64225', '64288', '64391', '64393', '64681', '64683', '64712', '6536', '6541', '6553', '6558', '6587', '6628', '6632', '6636', '72454', '74261', '84954', '85080', '85861', '85943', '85984', '87079', '87081', '87082', '87083', '87087', '87544', '87552', '87580', '87609', '87620', '87697', '88292', '88323', '88646', '88930', '90195', '90202', '90609', '90611', '90630', '92013', '92092', '92579', '92582', '92583', '92589', '92907

', '92947', '92950', '93273', '93274', '93606', '94161', '98100', '98138', '98197', '98262', '98283', '98560', '99008', '99021', '99206', '99207', '99773', '99780', '99781']
2024-05-31 08:03:17,201 - INFO - Epoch 1/50, Loss: 3.4904
2024-05-31 08:10:48,811 - INFO - Validation Loss: 2.2860, Accuracy: 0.4511
2024-05-31 08:10:49,727 - INFO - Validation loss decreased. Saving the model.
2024-05-31 08:25:42,985 - INFO - Epoch 2/50, Loss: 2.3342
2024-05-31 08:29:02,638 - INFO - Validation Loss: 1.8566, Accuracy: 0.5380
2024-05-31 08:29:03,571 - INFO - Validation loss decreased. Saving the model.
2024-05-31 08:43:39,079 - INFO - Epoch 3/50, Loss: 1.9911
2024-05-31 08:46:58,086 - INFO - Validation Loss: 1.7052, Accuracy: 0.5715
2024-05-31 08:46:59,011 - INFO - Validation loss decreased. Saving the model.
2024-05-31 09:01:41,496 - INFO - Epoch 4/50, Loss: 1.7915
2024-05-31 09:05:02,930 - INFO - Validation Loss: 1.6436, Accuracy: 0.5864
2024-05-31 09:05:03,856 - INFO - Validation loss decreased. Saving the model.
2024-05-31 09:19:44,017 - INFO - Epoch 5/50, Loss: 1.6635
2024-05-31 09:23:05,471 - INFO - Validation Loss: 1.5817, Accuracy: 0.6032
2024-05-31 09:23:06,409 - INFO - Validation loss decreased. Saving the model.
2024-05-31 09:37:48,824 - INFO - Epoch 6/50, Loss: 1.5753
2024-05-31 09:41:10,622 - INFO - Validation Loss: 1.5559, Accuracy: 0.6115
2024-05-31 09:41:11,549 - INFO - Validation loss decreased. Saving the model.
2024-05-31 09:55:49,847 - INFO - Epoch 7/50, Loss: 1.5074
2024-05-31 09:59:10,762 - INFO - Validation Loss: 1.5098, Accuracy: 0.6233
2024-05-31 09:59:11,689 - INFO - Validation loss decreased. Saving the model.
2024-05-31 10:13:51,085 - INFO - Epoch 8/50, Loss: 1.4442
2024-05-31 10:17:12,218 - INFO - Validation Loss: 1.5227, Accuracy: 0.6264
2024-05-31 10:17:12,218 - INFO - Validation loss did not decrease for 1 epochs.
2024-05-31 10:31:52,242 - INFO - Epoch 9/50, Loss: 1.3980
2024-05-31 10:35:10,872 - INFO - Validation Loss: 1.5776, Accuracy: 0.6203
2024-05-31 10:35:10,872 - INFO - Validation loss did not decrease for 2 epochs.
2024-05-31 10:49:55,541 - INFO - Epoch 10/50, Loss: 1.3532
2024-05-31 10:53:19,103 - INFO - Validation Loss: 1.5616, Accuracy: 0.6305
2024-05-31 10:53:19,103 - INFO - Validation loss did not decrease for 3 epochs.
2024-05-31 11:08:02,036 - INFO - Epoch 11/50, Loss: 1.3158
2024-05-31 11:11:23,746 - INFO - Validation Loss: 1.5306, Accuracy: 0.6338
2024-05-31 11:11:23,746 - INFO - Validation loss did not decrease for 4 epochs.
2024-05-31 11:26:04,401 - INFO - Epoch 12/50, Loss: 1.2879
2024-05-31 11:29:25,725 - INFO - Validation Loss: 1.6289, Accuracy: 0.6108
2024-05-31 11:29:25,725 - INFO - Validation loss did not decrease for 5 epochs.
2024-05-31 11:29:25,726 - INFO - Early stopping triggered. Stopping training.
2024-05-31 11:39:59,500 - INFO - Model training and conversion complete.
```

## Prediction Results

Below are some sample predictions made by the model along with the true labels.

```plaintext
Image: 52031_Light Stone Grey_1_1619493021.jpeg, True Label: 52031, Predicted Label: 52031
Image: 60599_Cool Yellow_0_1608731260.jpeg, True Label: 60599, Predicted Label: 60599
Image: 6005_Spring Yellowish Green_1_1619093773.jpeg, True Label: 6005, Predicted Label: 6005
Image: c0_3_zrjN_original_87697_1609708052902.jpg, True Label: 87697, Predicted Label: 87697
Image: 60479_Bright Purple_5_1619134784.jpeg, True Label: 60479, Predicted Label: 13731
Image: 41748_Dark Red_4_1621167171.jpeg, True Label: 41748, Predicted Label: 41747
Image: 60476_Lavender_3_1619498733.jpeg, True Label: 60476, Predicted Label: 4070
Image: 3002_Medium-Yellowish green_2_1619395102.jpeg, True Label: 3002, Predicted Label: 2357
Image: 14769_Medium Lilac_2_1621088525.jpeg, True Label: 14769, Predicted Label: 14769
Image: 60599_Olive Green_1_1608721562.jpeg, True Label: 60599, Predicted Label: 60599
Image: 64225_Dark Red_2_1619094201.jpeg, True Label: 64225, Predicted Label: 64225
Image: 57909_Reddish Brown_2_1621456661.jpeg, True Label: 57909b, Predicted Label: 90609
Image: 10197_Dark Brown_3_1621169005.jpeg, True Label: 10197, Predicted Label: 10288
Image: 16577_Flame Yellowish Orange_4_1619187470.jpeg, True Label: 16577, Predicted Label: 16577
Image: 3032_Sand Green_0_1621170824.jpeg, True Label: 3032, Predicted Label: 3032
Image: 6215_Bright Orange_0_1619563885.jpeg, True Label: 6215, Predicted Label: 3040b
Image: 3040a_Nougat_2_1619353406.jpeg, True Label: 3040b, Predicted Label: 60598
Image: c3_4_B5_BPzW_original_1618996976499.jpg, True Label: 13349, Predicted Label: 6232
Image: 47398_Medium Blue_2_1587370414.jpeg, True Label: 47398, Predicted Label: 47398
Image: 92583_Medium Lavender_0_1621459037.jpeg, True Label: 92583, Predicted Label: 92583
Image: c1_5_P49_NgxJ_original_1618572805933.jpg, True Label: 2540, Predicted Label: 2540
Image: 3707_Earth Blue_4_1619483413.jpeg, True Label: 3707, Predicted Label: 3707
Image: c3_5_mkxP_original_3020_1609710317561.jpg, True Label: 3020, Predicted Label: 3028
Image: 3623_Flame Yellowish Orange_0_1608687669.jpeg, True Label: 3623, Predicted Label: 3623
Image: 92903_Earth Green

_0_1619363844.jpeg, True Label: 6005, Predicted Label: 6005
Image: 6231_Cool Yellow_1_1619125022.jpeg, True Label: 6231, Predicted Label: 15712
Image: 63286_Light Nougat_3_1619549511.jpeg, True Label: 3069b, Predicted Label: 3069b
Image: 3676_Flame Yellowish Orange_5_1587566518.jpeg, True Label: 3676, Predicted Label: 41682
Image: 3942a_Medium-Yellowish green_2_1621105077.jpeg, True Label: 3942c, Predicted Label: 3942c
Image: 92950_White Glow_5_1619130179.jpeg, True Label: 92950, Predicted Label: 92950
Image: 11215_Light Royal Blue_2_1619504589.jpeg, True Label: 11215, Predicted Label: 11215
Image: 43719_Aqua_2_1619678891.jpeg, True Label: 43719, Predicted Label: 43719
Image: 6587_White_1_1621175126.jpeg, True Label: 6587, Predicted Label: 6587
Image: 20482_Cool Yellow_0_1621091188.jpeg, True Label: 20482, Predicted Label: 20482
Image: 60476_Cool Yellow_0_1621458294.jpeg, True Label: 60476, Predicted Label: 60476
Image: 23969_White_8_1588216333.jpeg, True Label: 23969, Predicted Label: 3040b
Image: c1_4_P11_SINK_original_1619000717772.jpg, True Label: 11153, Predicted Label: 11153
Image: 32123a_Light Nougat_2_1619493858.jpeg, True Label: 32123b, Predicted Label: 32123b
Image: 52216_Bright Purple_5_1619396910.jpeg, True Label: 41748, Predicted Label: 41748
Image: 3622_Bright Bluish Green_5_1619494165.jpeg, True Label: 3622, Predicted Label: 3659
Image: 30136_Sand Blue_4_1619561499.jpeg, True Label: 30136, Predicted Label: 36841
Image: 3665a_Medium Blue_0_1619123469.jpeg, True Label: 3665, Predicted Label: 3040b
Image: 87544_Aqua_0_1608786282.jpeg, True Label: 87544, Predicted Label: 87544
Image: 10314_White_1_1589300804.jpeg, True Label: 10314, Predicted Label: 11477
Image: 50305_Cool Yellow_1_1619460206.jpeg, True Label: 50305, Predicted Label: 50305
Image: 24299_Cool Yellow_1_1619493941.jpeg, True Label: 24299, Predicted Label: 24299
Image: 98100_Light Royal Blue_0_1619352361.jpeg, True Label: 98100, Predicted Label: 98100
Image: 32187_Nougat_1_1619159419.jpeg, True Label: 32187, Predicted Label: 74261
Image: 63868_Bright Green_1_1621081542.jpeg, True Label: 63868, Predicted Label: 63868
Image: 35382_Medium Azur_3_1619660508.jpeg, True Label: 3005, Predicted Label: 3005
```

Overall accuracy: 64.00%



### Model Performance and Evaluation

The model achieved an overall accuracy of 64.00% on the validation set. This performance indicates that while the model can correctly classify a significant number of LEGO brick images, there is room for improvement. The confusion matrix would provide more insights into which classes are being misclassified, and by analyzing this, further improvements can be targeted.

### Suggestions for Further Improvement

1. **Data Augmentation**: Implement more extensive data augmentation techniques such as random cropping, rotation, and flipping. This can help the model generalize better by exposing it to a wider variety of data.

2. **Model Architecture**: Experiment with more complex model architectures. Increasing the depth of the network or trying different types of layers (e.g., residual connections, inception modules) can improve performance.

3. **Hyperparameter Tuning**: Perform a thorough hyperparameter search to find the optimal learning rate, batch size, and other training parameters. Techniques like grid search or random search could be useful.

4. **Learning Rate Scheduling**: Implement learning rate schedules or adaptive learning rate methods like ReduceLROnPlateau, which decreases the learning rate when the validation performance plateaus.

5. **Regularization**: Increase regularization through techniques like dropout, L2 regularization, or adding noise to the input data. This helps prevent overfitting.

6. **Class Imbalance**: Address any class imbalance in the dataset by using techniques like class weighting, oversampling minority classes, or undersampling majority classes.

7. **Pre-trained Models**: Utilize transfer learning with pre-trained models such as ResNet, VGG, or EfficientNet. Fine-tuning these models on your dataset can significantly boost performance.

8. **Cross-Validation**: Use k-fold cross-validation to ensure the model's robustness and reliability across different subsets of the data.

9. **Additional Data**: If possible, gather more labeled data to increase the diversity and volume of the training set, which can help improve model performance.

### Performance Analysis

The current model shows that it can learn and distinguish between various classes of LEGO bricks to some extent. The validation accuracy suggests the model is reasonably effective but still struggles with certain classes, as evidenced by the validation loss not consistently decreasing. The early stopping mechanism helped in preventing overfitting, but it also highlights that the model might be hitting a performance ceiling with the current configuration.

By addressing the above suggestions, particularly through model architecture enhancements and data augmentation, further performance gains can be realized. This approach will likely result in a more accurate and robust classifier capable of handling the complexity and variability of real-world data. 
