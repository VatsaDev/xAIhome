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
import logging  #LOGGING

#LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        #LOGGING
        logging.info(f"SimpleNN initialized with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")

    def forward(self, x):
        #LOGGING
        logging.debug(f"Forward pass input shape: {x.shape}")
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #LOGGING
        logging.debug(f"Forward pass output shape: {x.shape}")
        return x

class VidDataset:
    def __init__(self, root_dir, clip_model, vae_model, device):
        self.root_dir = root_dir
        self.clip_model = clip_model
        self.vae_model = vae_model
        self.device = device
        #LOGGING
        logging.info(f"VidDataset initialized with root_dir={root_dir}")
        self.data = self.load_data()

    def load_data(self):
        #LOGGING
        logging.info("Loading data...")
        data = []
        videos = [f for f in os.listdir(self.root_dir) if f.endswith('.mp4')]
        #LOGGING
        logging.info(f"Found {len(videos)} videos")
        for video in videos:
            #LOGGING
            logging.info(f"Processing video: {video}")
            crops = self.process_video(video)
            data.extend(crops)
        #LOGGING
        logging.info(f"Total crops processed: {len(data)}")
        return data

    def process_video(self, video):
        #LOGGING
        logging.info(f"Processing video: {video}")
        crops = []
        
        cap = cv2.VideoCapture(os.path.join(self.root_dir, video))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        #LOGGING
        logging.info(f"Extracted {len(frames)} frames from video")

        frames = frames[:len(frames) - (len(frames) % 20)]
        frame_chunks = [frames[i:i + 20] for i in range(0, len(frames), 20)]
        frame_pairs = frame_chunks[:len(frame_chunks) - (len(frame_chunks) % 2)]
        #LOGGING
        logging.info(f"Created {len(frame_pairs)} frame pairs")
        
        vidstr = video.split("_")
        label = vidstr[0]
        #LOGGING
        logging.info(f"Video label: {label}")
        
        for i in range(0, len(frame_pairs), 2):
            video_pair = frame_pairs[i:i + 2]
            crops.extend(self.create_crops_from_pair(video_pair, label))

        #LOGGING
        logging.info(f"Created {len(crops)} crops from video")
        return crops

    def create_crops_from_pair(self, video_pair, label):
        #LOGGING
        logging.debug(f"Creating crops from pair with label: {label}")
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

            #LOGGING
            logging.debug(f"Extracting VAE features for crop 1")
            vae_f_1 = self.extract_vae_features(video_crops_1)
            #LOGGING
            logging.debug(f"Extracting CLIP features for label: {label}")
            clip_f = self.extract_clip_features(label)
            #LOGGING
            logging.debug(f"Extracting VAE features for crop 2")
            vae_f_2 = self.extract_vae_features(video_crops_2)

            crops.append((vae_f_1, clip_f, vae_f_2))

        #LOGGING
        logging.debug(f"Created {len(crops)} crops from pair")
        return crops

    def extract_vae_features(self, frames):
        #LOGGING
        logging.debug(f"Extracting VAE features from {len(frames)} frames")
        concatenated_frame = np.concatenate(frames, axis=1)
        pil_image = Image.fromarray(concatenated_frame)
        pil_image = pil_image.resize((256, 256))  # Resize to match VAE input size
        image_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            latent = self.vae_model.encode(image_tensor).latent_dist.sample()
        
        #LOGGING
        logging.debug(f"VAE features shape: {latent.squeeze(0).shape}")
        return latent.squeeze(0)

    def extract_clip_features(self, label):
        #LOGGING
        logging.debug(f"Extracting CLIP features for label: {label}")
        text_input = clip.tokenize([label]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
        #LOGGING
        logging.debug(f"CLIP features shape: {text_features.squeeze(0).shape}")
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
        #LOGGING
        logging.info(f"VidLoader initialized with batch_size={batch_size}, shuffle={shuffle}")

    def __iter__(self):
        indices = np.random.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            #LOGGING
            logging.debug(f"Yielding batch with indices: {batch_indices}")
            yield [self.dataset[idx] for idx in batch_indices]

    def __len__(self):
        return len(self.dataset) // self.batch_size

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    #LOGGING
    logging.info("Starting training...")
    for batch_idx, batch in enumerate(train_loader):
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
        #LOGGING
        logging.debug(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    #LOGGING
    logging.info(f"Training completed. Average loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    #LOGGING
    logging.info("Starting validation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            vae_f_1, clip_f, vae_f_2 = zip(*batch)
            vae_f_1 = torch.stack(vae_f_1).to(device)
            clip_f = torch.stack(clip_f).to(device)
            vae_f_2 = torch.stack(vae_f_2).to(device)
            
            output = model(torch.cat((vae_f_1.view(vae_f_1.size(0), -1), clip_f), dim=1))
            loss = criterion(output, vae_f_2.view(vae_f_2.size(0), -1))
            total_loss += loss.item()
            #LOGGING
            logging.debug(f"Validation Batch {batch_idx+1}/{len(val_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(val_loader)
    #LOGGING
    logging.info(f"Validation completed. Average loss: {avg_loss:.4f}")
    return avg_loss

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    #LOGGING
    logging.info("Starting testing...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            vae_f_1, clip_f, vae_f_2 = zip(*batch)
            vae_f_1 = torch.stack(vae_f_1).to(device)
            clip_f = torch.stack(clip_f).to(device)
            vae_f_2 = torch.stack(vae_f_2).to(device)
            
            output = model(torch.cat((vae_f_1.view(vae_f_1.size(0), -1), clip_f), dim=1))
            loss = criterion(output, vae_f_2.view(vae_f_2.size(0), -1))
            total_loss += loss.item()
            #LOGGING
            logging.debug(f"Test Batch {batch_idx+1}/{len(test_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(test_loader)
    #LOGGING
    logging.info(f"Testing completed. Average loss: {avg_loss:.4f}")
    return avg_loss

def main():
    #LOGGING
    logging.info("Starting main function")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #LOGGING
    logging.info(f"Using device: {device}")
    
    # Load CLIP model
    #LOGGING
    logging.info("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Load VAE model
    #LOGGING
    logging.info("Loading VAE model...")
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # Hyperparameters
    input_size = 4096 + 512  # 4096 (flattened vae_f_1) + 512 (clip_f)
    hidden_size = 2048
    output_size = 4096  # flattened vae_f_2 size
    lr = 0.001
    batch_size = 32
    num_epochs = 2
    #LOGGING
    logging.info(f"Hyperparameters: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}")
    
    # Load and split dataset
    #LOGGING
    logging.info("Loading and splitting dataset...")
    dataset = VidDataset("vids", clip_model, vae_model, device)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = VidLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = VidLoader(val_dataset, batch_size=batch_size)
    test_loader = VidLoader(test_dataset, batch_size=batch_size)
    #LOGGING
    logging.info(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Initialize model, optimizer, and loss function
    #LOGGING
    logging.info("Initializing model, optimizer, and loss function...")
    model = SimpleNN(input_size, hidden_size, output_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()
    
    # Training loop
    #LOGGING
    logging.info("Starting training loop...")
    for epoch in range(num_epochs):
        #LOGGING
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        test_loss = test(model, test_loader, criterion, device)
        
        #LOGGING
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        scheduler.step()
    
    #LOGGING
    logging.info("Training completed")
    
if __name__ == "__main__":
    main()