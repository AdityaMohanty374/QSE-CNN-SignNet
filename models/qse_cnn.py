import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import os
from tqdm import tqdm
import warnings
import pennylane as qml

warnings.filterwarnings('ignore')


class QSEBlock(nn.Module):
    """
    Quantum Squeeze-and-Excitation Block
    Paper: "Quantum Squeeze-and-Excitation Networks" (Peng et al., 2024)
    
    Replaces the first FC layer of classical SE with quantum computing:
    1. Squeezing: Global average pooling
    2. Quantum Excitation: Amplitude embedding + CNOT gates for dimension reduction
    3. Classical Excitation: Second FC layer + Sigmoid
    4. Scaling: Multiply features by learned channel weights
    """
    def __init__(self, channels, reduction=16):
        super(QSEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.reduced_channels = channels // reduction
        
        # Squeeze operation
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Calculate number of qubits needed
        self.n_qubits = int(np.ceil(np.log2(channels)))
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        # Create quantum circuit as a QNode
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def quantum_circuit(inputs, output_idx):
            """
            Quantum circuit for channel reduction
            Args:
                inputs: Normalized input features
                output_idx: Index of output channel to measure
            """
            # Amplitude embedding
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_qubits), 
                                   normalize=True, pad_with=0.0)
            
            # Apply CNOT gates to create entanglement between adjacent qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Measure Pauli-Z on the appropriate qubit
            qubit_to_measure = output_idx % self.n_qubits
            return qml.expval(qml.PauliZ(qubit_to_measure))
        
        self.quantum_circuit = quantum_circuit
        
        # Second fully connected layer (classical)
        self.fc2 = nn.Linear(self.reduced_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def quantum_excitation(self, z):
        """
        Apply quantum computing to perform dimension reduction
        Args:
            z: Input tensor of shape (batch_size, channels)
        Returns:
            Tensor of shape (batch_size, reduced_channels)
        """
        batch_size = z.size(0)
        u1 = torch.zeros(batch_size, self.reduced_channels, device=z.device, dtype=z.dtype)
        
        for b in range(batch_size):
            # Get single sample and pad to 2^n_qubits if necessary
            sample = z[b].detach().cpu().numpy()
            n_features_needed = 2 ** self.n_qubits
            
            if len(sample) < n_features_needed:
                sample = np.pad(sample, (0, n_features_needed - len(sample)), 
                               mode='constant', constant_values=0.0)
            elif len(sample) > n_features_needed:
                sample = sample[:n_features_needed]
            
            # Compute quantum measurements for each output channel
            for j in range(self.reduced_channels):
                measurement = self.quantum_circuit(sample, j)
                u1[b, j] = measurement
            
        return u1
    
    def forward(self, x):
        """
        Forward pass of QSE block
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Recalibrated tensor of same shape as input
        """
        b, c, _, _ = x.size()
        
        # Squeeze: Global information embedding
        z = self.squeeze(x).view(b, c)
        
        # Quantum Excitation: Dimension reduction via quantum computing
        u1 = self.quantum_excitation(z)
        
        # Classical Excitation: Second FC layer
        y = self.fc2(u1)
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        # Scale: Channel-wise multiplication
        return x * y.expand_as(x)


class SignLanguageCNN(nn.Module):
    """
    Enhanced CNN model with QSE blocks for improved sign language recognition.
    QSE blocks use quantum computing to enhance feature recalibration while
    reducing parameters compared to classical SE blocks.
    """
    def __init__(self, num_classes=26, input_channels=3, dropout_rate=0.3, se_reduction=16):
        super(SignLanguageCNN, self).__init__()
        
        # Feature extraction layers with QSE blocks
        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.qse1 = QSEBlock(32, reduction=se_reduction)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.qse2 = QSEBlock(64, reduction=se_reduction)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.qse3 = QSEBlock(128, reduction=se_reduction)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.qse4 = QSEBlock(256, reduction=se_reduction)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Calculate the size of flattened features
        # For input size 224x224: after 4 pooling layers -> 14x14
        self.fc_input_size = 256 * 14 * 14
        
        # Classification layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Feature extraction with QSE blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.qse1(x)  # Apply QSE block
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.qse2(x)  # Apply QSE block
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.qse3(x)  # Apply QSE block
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.qse4(x)  # Apply QSE block
        x = self.pool4(x)
        
        # Flatten for classification
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class SignLanguageDataset(Dataset):
    """
    Custom dataset class for sign language images with preprocessing.
    """
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
    
    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(class_dir, filename)
                        samples.append((path, self.class_to_idx[class_name]))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        
        try:
            with open(path, 'rb') as f:
                image = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target


def get_data_transforms():
    """
    Enhanced data augmentation for training and validation.
    """
    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=20),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
    ])
    
    # Validation/Test transforms without augmentation
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


class SignLanguageTrainer:
    """
    Training class for QSE-enhanced sign language model.
    """
    def __init__(self, model, device, num_classes=26):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001, weight_decay=5e-4):
        # Use label smoothing to reduce overfitting
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # AdamW optimizer with better weight decay handling
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Cosine annealing with warm restarts for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 25
        
        print("Starting training with QSE-enhanced CNN...")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation phase
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies
                }, '/kaggle/working/best_qse_sign_language_model.pth')
                print(f"✓ New best model saved! Validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
        
        return val_preds, val_targets
    
    def plot_training_history(self):
        """Plot training and validation metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss', color='blue', linewidth=2)
        ax1.plot(self.val_losses, label='Validation Loss', color='red', linewidth=2)
        ax1.set_title('Training and Validation Loss (with QSE Blocks)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red', linewidth=2)
        ax2.set_title('Training and Validation Accuracy (with QSE Blocks)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qse_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix (QSE-CNN)'):
    """Plot confusion matrix with proper formatting."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('qse_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to run the QSE-enhanced sign language recognition system.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    NUM_CLASSES = 26
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    DROPOUT_RATE = 0.3
    SE_REDUCTION = 16
    
    # Create QSE-enhanced model
    model = SignLanguageCNN(
        num_classes=NUM_CLASSES, 
        dropout_rate=DROPOUT_RATE,
        se_reduction=SE_REDUCTION
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"QSE blocks integrated for quantum-enhanced feature recalibration")
    
    # Get transforms
    train_transforms, val_transforms = get_data_transforms()
    
    # Dataset path
    data_path = "/kaggle/input/signlang-dataset"
    
    if os.path.exists(data_path):
        # Load dataset
        full_dataset = SignLanguageDataset(data_path, transform=train_transforms)
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Apply different transforms to validation set
        val_dataset.dataset.transform = val_transforms
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create trainer
        trainer = SignLanguageTrainer(model, device, NUM_CLASSES)
        
        # Train model
        val_preds, val_targets = trainer.train(
            train_loader, val_loader, 
            num_epochs=NUM_EPOCHS, 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        
        # Plot training history
        trainer.plot_training_history()
        
        # Create confusion matrix
        class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        plot_confusion_matrix(val_targets, val_preds, class_names)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(val_targets, val_preds, target_names=class_names))
        
    else:
        print(f"Dataset path '{data_path}' not found.")
        print("\nTo use this implementation:")
        print("1. Organize your sign language images by letter (A-Z folders)")
        print("2. Update the 'data_path' variable with your dataset location")
        print("3. Install PennyLane: pip install pennylane")
        print("4. Run the script again")


class SignLanguageInference:
    """
    Inference class for QSE-CNN model.
    """
    def __init__(self, model_path, device):
        self.device = device
        self.model = SignLanguageCNN(num_classes=26, dropout_rate=0.3)
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Class names
        self.class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Predict the sign language letter from an image."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        predicted_letter = self.class_names[predicted_class.item()]
        confidence_score = confidence.item()
        
        return predicted_letter, confidence_score


if __name__ == "__main__":
    main()
