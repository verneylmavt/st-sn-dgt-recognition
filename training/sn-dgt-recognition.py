#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import os
import random


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# In[4]:


import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# #### Seed Setting

# ```markdown
# In here, the code sets the random seed for reproducibility across random, NumPy, and PyTorch operations. This ensures consistent results by fixing the seed for both CPU and GPU computations.
# ```

# In[6]:


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()


# #### Data Transformation

# ```markdown
# In here, the code handles the data preprocessing by applying transformations such as random rotation, affine transformations, and normalization for training data, while only applying normalization for test data. These transformations enhance generalization and standardize the pixel values.
# ```

# In[7]:


train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# In[8]:


test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# #### Data Loading

# ```markdown
# In here, the code loads the MNIST dataset using the datasets library provided by torchvision.
# ```

# In[9]:


train_val_dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=train_transforms)
test_dataset = datasets.MNIST(root='../../data', train=False, download=True, transform=test_transforms)


# #### Dataset and DataLoader

# ```markdown
# In here, the code splits the training data into training and validation subsets, initializes data loaders for training, validation, and testing datasets, and specifies batch sizes for efficient data iteration.
# ```

# In[10]:


val_size = 10000
train_size = len(train_val_dataset) - val_size


# In[11]:


train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])


# In[12]:


batch_size = 128

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# #### Data Visualization

# ```markdown
# In here, the code provides a function for visualizing images from the dataset with an option to save the displayed images. It allows filtering by label and displaying a specified number of images.
# ```

# In[13]:


def show_images(dataset, label_to_display=None, num_images=1, save_images=False):
    output_dir = "../data/"
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 2))
    count = 0
    
    for image, label in dataset:
        if label_to_display is not None and label != label_to_display:
            continue
        
        image = image.squeeze().numpy()
        plt.subplot(1, num_images, count + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
        
        if save_images:
            save_path = os.path.join(output_dir, f"label_{label}_image_{count}.png")
            plt.imsave(save_path, image, cmap='gray')
        
        count += 1
        if count >= num_images:
            break
    
    plt.show()


# In[14]:


# show_images(train_dataset, label_to_display=0, num_images=1, save_images=True)


# #### Model Definition

# ```markdown
# In here, the code defines a Convolutional Neural Network (CNN) for digit classification, utilizing residual blocks for feature extraction and pooling layers for dimensionality reduction. The final layers include global average pooling, dropout for regularization, and a fully connected layer for classification.
# 
# • Residual Block
# The inclusion of residual blocks ensures efficient training by allowing the network to learn identity mappings, addressing issues like vanishing gradients. This is particularly useful in deeper networks, where learning residual functions (the difference between the input and output) is easier than learning the full mapping directly. By incorporating these blocks, the model achieves better performance with fewer training epochs.
# 
# • Layer (Layer 1 - Layer 6)
# - Each layer pair (e.g., layer1 + layer2, layer3 + layer4, etc.) extracts features at increasing levels of abstraction.
# - The initial layers capture basic patterns like edges and textures, while deeper layers focus on complex patterns, such as shapes and digit structures.
# - Residual blocks allow these layers to preserve essential information from earlier layers while refining feature representations.
# 
# • Max Pooling
# The pooling layers (pool1, pool2, pool3) progressively downsample feature maps, reducing spatial dimensions while retaining critical information. This operation not only reduces computational complexity but also introduces translational invariance, which is essential for digit classification.
# 
# • Global Average Pooling
# The adaptive average pooling layer aggregates spatial information into a single 1 × 1 feature map per channel. This approach reduces the feature map to a fixed size, independent of the input dimensions, making the architecture more versatile for inputs of varying sizes. By summarizing the feature maps, global average pooling reduces the risk of overfitting compared to fully connected layers with large weight matrices.
# 
# • Fully Connected Layer
# The final linear layer maps the aggregated features to the output dimension (num_classes), which corresponds to the number of possible digit classes (e.g., 10 for digits 0–9).
# This layer produces logits, which can be converted into probabilities using a softmax function during evaluation or training.
# ```

# In[15]:


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # First Convolutional Layer for Residual Block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        # Batch Normalization Layer for First Convolution
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Activation Layer for Non-Linear Transformation
        self.relu = nn.ReLU(inplace=True)
        # Second Convolutional Layer for Residual Block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        # Batch Normalization Layer for Second Convolution
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsampling Layer for Adjusting Identity Shortcut
        self.downsample = downsample
    
    def forward(self, x):
        # Identity Shortcut Connection
        identity = x
        
        # First Convolutional Layer Transformation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Second Convolutional Layer Transformation
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Downsampling of Identity Shortcut (if Applicable)
        if self.downsample:
            identity = self.downsample(x)
        # Addition of Identity Shortcut to Residual Block Output
        out += identity
        # Activation of Residual Block Output
        out = self.relu(out)
        return out


# In[16]:


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # First Convolutional Layer for Feature Extraction
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # First Residual Block for Feature Refinement
        self.layer2 = ResidualBlock(32, 32)
        # First Max Pooling Layer for Spatial Downsampling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer for Feature Extraction
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Second Residual Block for Feature Refinement
        self.layer4 = ResidualBlock(64, 64)
        # Second Max Pooling Layer for Spatial Downsampling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third Convolutional Layer for Feature Extraction
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Third Residual Block for Feature Refinement
        self.layer6 = ResidualBlock(128, 128)
        # Third Max Pooling Layer for Spatial Downsampling
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling Layer for Spatial Dimension Reduction
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout Layer for Regularization
        self.dropout = nn.Dropout(0.5)
        # Fully Connected Layer for Digit Classification
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # First Convolutional Layer Transformation
        out = self.layer1(x)
        # First Residual Block Transformation
        out = self.layer2(out)
        # First Max Pooling Layer Transformation
        out = self.pool1(out)
        
        # Second Convolutional Layer Transformation
        out = self.layer3(out)
        # Second Residual Block Transformation
        out = self.layer4(out)
        # Second Max Pooling Layer Transformation
        out = self.pool2(out)
        
        # Third Convolutional Layer Transformation
        out = self.layer5(out)
        # Third Residual Block Transformation
        out = self.layer6(out)
        # Third Max Pooling Layer Transformation
        out = self.pool3(out)
        
        # Global Average Pooling Transformation
        out = self.global_avg_pool(out)
        # Flattening of Pooled Output
        out = out.view(out.size(0), -1)
        # Dropout Application to Flattened Output
        out = self.dropout(out)
        # Transformation of Flattened Features → Digit Scores
        out = self.fc(out)
        return out


# #### Training Function

# ```markdown
# In here, the code defines a function to train the model for one epoch. It processes each batch, performs backpropagation, updates the model parameters, and calculates the training accuracy and loss.
# ```

# In[17]:


def train_epoch(net, iter, criterion, optimizer):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in iter:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = net(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(iter.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# #### Evaluation Function

# ```markdown
# In here, the code defines a function to evaluate the model on a validation set. It computes the loss and accuracy without updating the model parameters.
# ```

# In[18]:


def evaluate_epoch(net, iter, criterion):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in iter:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(iter.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# #### Training

# ```markdown
# In here, the code trains the model, monitors performance metrics, and visualizes the training and validation losses and accuracies over epochs.
# ```

# In[19]:


def train_model(net, train_iter, val_iter, criterion, optimizer, scheduler, num_epochs=20, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    num_epochs_used = 0
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nepoch {epoch}/{num_epochs}")
        num_epochs_used += 1
        train_loss, train_acc = train_epoch(net, train_iter, criterion, optimizer)
        val_loss, val_acc = evaluate_epoch(net, val_iter, criterion)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(net.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    
    return train_losses, train_accuracies, val_losses, val_accuracies, num_epochs_used


# In[20]:


net = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs = 50
patience = 5


# In[21]:


train_losses, train_accuracies, val_losses, val_accuracies, num_epochs_used = train_model(net, train_iter, val_iter, criterion, optimizer, scheduler, num_epochs, patience)


# In[22]:


epochs_range = range(1, num_epochs_used + 1)
plt.figure(figsize=(6, 4))

plt.plot(epochs_range, train_losses, label='train loss', linestyle='-', color='#2a7db8')
plt.plot(epochs_range, train_accuracies, label='train acc', linestyle='--', color='green')
plt.plot(epochs_range, val_losses, label='val loss', linestyle='-', color='red')
plt.plot(epochs_range, val_accuracies, label='val acc', linestyle='--', color='magenta')
plt.xlabel('epoch')
plt.ylabel('value')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

plt.show()


# #### Evaluation Metrics

# ```markdown
# In here, the code defines the cal_metrics function, which evaluates the trained model on the test dataset. It sets the model to evaluation mode, iterates through the test data loader, makes predictions, and accumulates the true and predicted labels. It then computes and prints the classification report (precision, recall, f1-score, support) for each digit.
# ```

# In[23]:


net.load_state_dict(torch.load('best_model.pth'))


# In[24]:


def cal_metrics(net, test_iter):
    net.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_iter:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds)
    print("\nClassification Report:\n", report)
    
    return None


# In[25]:


cal_metrics(net, test_iter)


# #### ONNX Exporting

# In[26]:


device = torch.device("cpu")

net.to(device)
net.eval()

dummy_input = torch.randn(1, 1, 28, 28, device=device)
torch.onnx.export(net, dummy_input, "model.onnx", verbose=True,
                    input_names=['input'], output_names=['output'], 
                    opset_version=11)


# In[28]:


# quantize_dynamic("model.onnx", "model-q.onnx", weight_type=QuantType.QInt8)

