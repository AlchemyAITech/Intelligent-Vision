import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

# --- 2. 神经网络模型定义 (PyTorch) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, layer_configs=[16, 32]):
        super(SimpleCNN, self).__init__()
        self.layer_configs = layer_configs
        self.num_layers = len(layer_configs)
        self.features = nn.ModuleList()
        
        in_channels = 1
        
        for out_channels in layer_configs:
            self.features.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.features.append(nn.ReLU())
            self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            
        # Calculate FC Input Size
        # MNIST 28x28 -> [pool] -> 14x14 -> [pool] -> 7x7 ...
        final_size = 28
        for _ in range(self.num_layers):
            final_size = final_size // 2
        
        if final_size < 1: final_size = 1
        
        self.fc_input_dim = in_channels * final_size * final_size
        
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def get_features(self, x):
        features = {}
        # Manually iterate to capture intermediate outputs
        # Structure: Conv, ReLU, Pool, Conv, ReLU, Pool ...
        
        current_x = x
        layer_idx = 0
        
        # features list contains: [Conv2d, ReLU, MaxPool2d, Conv2d, ReLU, MaxPool2d, ...]
        # We want to capture Output of Conv and Pool
        
        for i, layer in enumerate(self.features):
            current_x = layer(current_x)
            
            if isinstance(layer, nn.Conv2d):
                layer_idx += 1
                features[f'Conv{layer_idx}'] = current_x
            elif isinstance(layer, nn.MaxPool2d):
                features[f'Pool{layer_idx}'] = current_x
                
        return features

def tensor_to_img_array(tensor):
    """
    Convert PyTorch tensor (C, H, W) or (H, W) to numpy array for display
    Normlized [-1, 1] -> [0, 1] -> [0, 255]
    """
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0] # Take first channel if multiple (usually feature map is 1 channel viewed)
        
    # Stats for normalization
    min_v, max_v = arr.min(), arr.max()
    if max_v - min_v > 1e-5:
        arr = (arr - min_v) / (max_v - min_v) * 255
    else:
        arr = np.zeros_like(arr)
        
    return arr.astype(np.uint8)

# --- 数据加载辅助 ---
def load_mnist_data(root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        # Download=True checks if exists
        train_full = datasets.MNIST(root, train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root, train=False, download=True, transform=transform)
        
        # Split train into train/val if needed, or just use slice for speed
        return train_full, test_data
    except Exception as e:
        return None, None

def get_random_sample(dataset):
    idx = np.random.randint(0, len(dataset))
    img, label = dataset[idx]
    return img.unsqueeze(0), label # Add batch dim

def calculate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total
