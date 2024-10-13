import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Custom dataset class
class VideoFrameDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Function to process video clips and create dataset
def process_video_clips(input_folder, output_folder):
    logging.info(f"Processing video clips from {input_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output folder: {output_folder}")

    dataset = []
    class_names = set()

    for filename in tqdm(os.listdir(input_folder), desc="Processing videos"):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Add more video formats if needed
            video_path = os.path.join(input_folder, filename)
            label = filename.split('_')[0]
            class_names.add(label)

            cap = cv2.VideoCapture(video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            if frames:
                combined_frame = np.hstack(frames)
                output_filename = f"{filename[:-4]}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, combined_frame)

                dataset.append((output_path, label))
                logging.info(f"Processed {filename} - Label: {label}")

    logging.info(f"Processed {len(dataset)} videos with {len(class_names)} unique classes")
    return dataset, list(class_names)

# Main training function
def train_cnn(input_folder, output_folder):
    logging.info("Starting CNN training process")
    
    # Process video clips
    dataset, class_names = process_video_clips(input_folder, output_folder)
    num_classes = len(class_names)
    logging.info(f"Number of classes: {num_classes}")

    # Split dataset into train and validation sets
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
    logging.info(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    train_dataset = VideoFrameDataset(train_data, transform=transform)
    val_dataset = VideoFrameDataset(val_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs = inputs.to(device)
            labels = torch.tensor([class_names.index(label) for label in labels]).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                avg_loss = total_loss / 10
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Avg Loss: {avg_loss:.4f}")
                total_loss = 0

                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_inputs, val_labels in val_loader:
                        val_inputs = val_inputs.to(device)
                        val_labels = torch.tensor([class_names.index(label) for label in val_labels]).to(device)
                        val_outputs = model(val_inputs)
                        val_loss += criterion(val_outputs, val_labels).item()
                    val_loss /= len(val_loader)
                    logging.info(f"Validation Loss: {val_loss:.4f}")
                model.train()

    logging.info("Training completed!")

# Example usage
input_folder = "clips2"
output_folder = "img"
train_cnn(input_folder, output_folder)