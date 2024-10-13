# This is VAE feature example code, so it differs from being a pure video loader

import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
from torch.optim.lr_scheduler import StepLR
import clip
from PIL import Image
from diffusers import AutoencoderKL
from datasets import load_dataset
import requests
from tqdm import tqdm

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class VidDataset:
    def __init__(self, dataset_name, clip_model, vae_model, device):
        self.dataset_name = dataset_name
        self.clip_model = clip_model
        self.vae_model = vae_model
        self.device = device
        self.data = self.load_data()

    def load_data(self):
        # Create 'vids' directory if it doesn't exist
        os.makedirs('vids', exist_ok=True)

        # Load the dataset from Hugging Face
        dataset = load_dataset(self.dataset_name, split="train")

        # Download videos
        for item in tqdm(dataset, desc="Downloading videos"):
            video_url = item['video']
            video_filename = os.path.join('vids', f"{item['id']}.mp4")
            
            if not os.path.exists(video_filename):
                response = requests.get(video_url, stream=True)
                with open(video_filename, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

        # Process downloaded videos
        data = []
        for video_file in os.listdir('vids'):
            if video_file.endswith('.mp4'):
                crops = self.process_video(video_file)
                data.extend(crops)

        return data

    def process_video(self, video):
        crops = []
        
        cap = cv2.VideoCapture(os.path.join(self.root_dir, video))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        frames = frames[:len(frames) - (len(frames) % 20)]
        frame_chunks = [frames[i:i + 20] for i in range(0, len(frames), 20)]
        frame_pairs = frame_chunks[:len(frame_chunks) - (len(frame_chunks) % 2)]
        
        vidstr = video.split("_")
        label = vidstr[0]
        
        for i in range(0, len(frame_pairs), 2):
            video_pair = frame_pairs[i:i + 2]
            crops.extend(self.create_crops_from_pair(video_pair, label))

        return crops

    def create_crops_from_pair(self, video_pair, label):
        crops = []
        for _ in range(3):
            crop_x, crop_y = np.random.randint(0, 768-256), np.random.randint(0, 768-256)
            video_crops_1, video_crops_2 = [], []

            for frames in video_pair:
                for frame in frames:
                    crop_1 = frame[crop_y:crop_y + 256, crop_x:crop_x + 256]
                    crop_2 = frame[crop_y:crop_y + 256, crop_x:crop_x + 256]
                    video_crops_1.append(crop_1)
                    video_crops_2.append(crop_2)

            vae_f_1 = self.extract_vae_features(video_crops_1)
            clip_f = self.extract_clip_features(label)
            vae_f_2 = self.extract_vae_features(video_crops_2)

            crops.append((vae_f_1, clip_f, vae_f_2))

        return crops

    def extract_vae_features(self, frames):
        concatenated_frame = np.concatenate(frames, axis=1)
        pil_image = Image.fromarray(concatenated_frame)
        pil_image = pil_image.resize((256, 256))  # Resize to match VAE input size
        image_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            latent = self.vae_model.encode(image_tensor).latent_dist.sample()
        
        return latent.squeeze(0)

    def extract_clip_features(self, label):
        text_input = clip.tokenize([label]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
        return text_features.squeeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class VidLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.random.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            yield [self.dataset[idx] for idx in batch_indices]

    def __len__(self):
        return len(self.dataset) // self.batch_size

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        vae_f_1, clip_f, vae_f_2 = zip(*batch)
        vae_f_1 = torch.stack(vae_f_1).to(device)
        clip_f = torch.stack(clip_f).to(device)
        vae_f_2 = torch.stack(vae_f_2).to(device)
        
        optimizer.zero_grad()
        output = model(torch.cat((vae_f_1.view(vae_f_1.size(0), -1), clip_f), dim=1))
        loss = criterion(output, vae_f_2.view(vae_f_2.size(0), -1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            vae_f_1, clip_f, vae_f_2 = zip(*batch)
            vae_f_1 = torch.stack(vae_f_1).to(device)
            clip_f = torch.stack(clip_f).to(device)
            vae_f_2 = torch.stack(vae_f_2).to(device)
            
            output = model(torch.cat((vae_f_1.view(vae_f_1.size(0), -1), clip_f), dim=1))
            loss = criterion(output, vae_f_2.view(vae_f_2.size(0), -1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            vae_f_1, clip_f, vae_f_2 = zip(*batch)
            vae_f_1 = torch.stack(vae_f_1).to(device)
            clip_f = torch.stack(clip_f).to(device)
            vae_f_2 = torch.stack(vae_f_2).to(device)
            
            output = model(torch.cat((vae_f_1.view(vae_f_1.size(0), -1), clip_f), dim=1))
            loss = criterion(output, vae_f_2.view(vae_f_2.size(0), -1))
            total_loss += loss.item()
    return total_loss / len(test_loader)

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Load VAE model
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # Hyperparameters
    input_size = 4096 + 512  # 4096 (flattened vae_f_1) + 512 (clip_f)
    hidden_size = 2048
    output_size = 4096  # flattened vae_f_2 size
    lr = 0.001
    batch_size = 32
    num_epochs = 2
    
    # Load and split dataset
    # Replace 'your_dataset_name' with the actual name of the Hugging Face dataset you want to use
    dataset = VidDataset("VatsaDev/Sportsvid", clip_model, vae_model, device)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = VidLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = VidLoader(val_dataset, batch_size=batch_size)
    test_loader = VidLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model, optimizer, and loss function
    model = SimpleNN(input_size, hidden_size, output_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        test_loss = test(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        scheduler.step()
    
if __name__ == "__main__":
    main()