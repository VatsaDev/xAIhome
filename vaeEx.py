import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.optim.lr_scheduler import StepLR
import clip
from PIL import Image
from diffusers import AutoencoderKL
import logging
import multiprocessing
from functools import partial
import time
import subprocess
import tempfile
import shutil
from tqdm import tqdm

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        logging.info(f"SimpleNN initialized with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")

    def forward(self, x):
        logging.debug(f"Forward pass input shape: {x.shape}")
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        logging.debug(f"Forward pass output shape: {x.shape}")
        return x

def process_video(video, root_dir, output_dir):
    try:
        video_path = os.path.join(root_dir, video)
        vidstr = video.split("_")
        label = vidstr[0]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames at 4 fps
            extract_cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", "fps=4",
                "-qmin", "1", "-qmax", "1", "-q:v", "1",
                f"{temp_dir}/frame%04d.png"
            ]
            subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            frames = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
            frames = frames[:len(frames) - (len(frames) % 20)]  # Ensure divisible by 20 (5 seconds * 4 fps)
            
            clips = []
            for pair_index in range(len(frames) // 20):
                start_idx = pair_index * 20
                frame_pair = frames[start_idx:start_idx + 20]
                clips.extend(create_crops_from_pair(frame_pair, label, pair_index, temp_dir, output_dir))
        
        return clips
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing video {video}: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error processing video {video}: {str(e)}")
        return []

def create_crops_from_pair(frame_pair, label, pair_index, temp_dir, output_dir):
    clips = []
    for crop_index in range(3):
        try:
            crop_x, crop_y = np.random.randint(0, 768-256), np.random.randint(0, 768-256)
            
            clip_name = f"{label}_pair_{pair_index}_crop_{crop_index}.mp4"
            output_path = os.path.join(output_dir, clip_name)
            
            create_clip(frame_pair, crop_x, crop_y, output_path, temp_dir)
            
            clips.append(clip_name)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating crop {crop_index} for {label}_pair_{pair_index}: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error creating crop {crop_index} for {label}_pair_{pair_index}: {str(e)}")
    
    return clips

def create_clip(frames, crop_x, crop_y, output_path, temp_dir):
    input_file = os.path.join(temp_dir, "input.txt")
    with open(input_file, "w") as f:
        for frame in frames:
            f.write(f"file '{os.path.join(temp_dir, frame)}'\n")
            f.write("duration 0.25\n")  # 4 fps
    
    ffmpeg_cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", input_file,
        "-filter:v", f"crop=256:256:{crop_x}:{crop_y}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-y", output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def create_clips(root_dir, output_dir):
    logging.info("Starting video clip creation...")
    os.makedirs(output_dir, exist_ok=True)
    
    videos = [f for f in os.listdir(root_dir) if f.endswith('.mp4')]
    logging.info(f"Found {len(videos)} videos")
    
    num_workers = multiprocessing.cpu_count()
    logging.info(f"Using {num_workers} workers for multiprocessing")
    
    with multiprocessing.Pool(num_workers) as pool:
        process_func = partial(process_video, root_dir=root_dir, output_dir=output_dir)
        all_clips = list(tqdm(pool.imap(process_func, videos), total=len(videos), desc="Processing videos"))
    
    all_clips = [clip for sublist in all_clips for clip in sublist]
    logging.info(f"Total clips created: {len(all_clips)}")
    return all_clips

class VidDataset:
    def __init__(self, clips_dir):
        self.clips_dir = clips_dir
        logging.info(f"VidDataset initialized with clips_dir={clips_dir}")
        self.data = self.load_data()

    def load_data(self):
        logging.info("Loading data...")
        data = []
        clips = [f for f in os.listdir(self.clips_dir) if f.endswith('.mp4')]
        logging.info(f"Found {len(clips)} clips")
        
        for i in range(0, len(clips), 2):
            if i + 1 < len(clips):
                clip_pair = (clips[i], clips[i+1])
                label = clips[i].split("_")[0]  # Assuming label is the first part of the filename
                data.append((clip_pair, label))
        
        logging.info(f"Total clip pairs: {len(data)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_pair, label = self.data[idx]
        return clip_pair, label

class VidLoader:
    def __init__(self, dataset, clip_model, vae_model, device, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.clip_model = clip_model
        self.vae_model = vae_model
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        logging.info(f"VidLoader initialized with batch_size={batch_size}, shuffle={shuffle}")

    def __iter__(self):
        indices = np.random.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.process_batch(batch)

    def process_batch(self, batch):
        clip_pairs, labels = zip(*batch)
        vae_f_1 = []
        vae_f_2 = []
        clip_f = []
        
        for (clip_1, clip_2), label in zip(clip_pairs, labels):
            vae_f_1.append(self.extract_vae_features(os.path.join(self.dataset.clips_dir, clip_1)))
            vae_f_2.append(self.extract_vae_features(os.path.join(self.dataset.clips_dir, clip_2)))
            clip_f.append(self.extract_clip_features(label))
        
        return torch.stack(vae_f_1), torch.stack(clip_f), torch.stack(vae_f_2)

    def extract_vae_features(self, clip_path):
        cap = cv2.VideoCapture(clip_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
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
        return len(self.dataset) // self.batch_size

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    logging.info(f"Starting training for epoch {epoch}...")
    for batch_idx, (vae_f_1, clip_f, vae_f_2) in enumerate(train_loader):
        vae_f_1 = vae_f_1.to(device)
        clip_f = clip_f.to(device)
        vae_f_2 = vae_f_2.to(device)
        
        optimizer.zero_grad()
        output = model(torch.cat((vae_f_1.view(vae_f_1.size(0), -1), clip_f), dim=1))
        loss = criterion(output, vae_f_2.view(vae_f_2.size(0), -1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        logging.info(f"Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    logging.info("Starting validation...")
    with torch.no_grad():
        for batch_idx, (vae_f_1, clip_f, vae_f_2) in enumerate(val_loader):
            vae_f_1 = vae_f_1.to(device)
            clip_f = clip_f.to(device)
            vae_f_2 = vae_f_2.to(device)
            
            output = model(torch.cat((vae_f_1.view(vae_f_1.size(0), -1), clip_f), dim=1))
            loss = criterion(output, vae_f_2.view(vae_f_2.size(0), -1))
            total_loss += loss.item()
            logging.debug(f"Validation Batch {batch_idx+1}/{len(val_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(val_loader)
    logging.info(f"Validation completed. Average loss: {avg_loss:.4f}")
    return avg_loss

def main():
    logging.info("Starting main function")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create clips
    root_dir = "vids"
    output_dir = "clips"
    all_clips = create_clips(root_dir, output_dir)
    
    # Load CLIP model
    logging.info("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Load VAE model
    logging.info("Loading VAE model...")
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # Hyperparameters
    input_size = 4096 + 512  # 4096 (flattened vae_f_1) + 512 (clip_f)
    hidden_size = 2048
    output_size = 4096  # flattened vae_f_2 size
    lr = 0.001
    batch_size = 32
    num_epochs = 10
    logging.info(f"Hyperparameters: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}")
    
    # Initialize model, optimizer, and loss function
    logging.info("Initializing model, optimizer, and loss function...")
    model = SimpleNN(input_size, hidden_size, output_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()
    
    # Training loop
    logging.info("Starting training loop...")
    for epoch in range(num_epochs):
        # Create new dataset and loaders for each epoch
        dataset = VidDataset(output_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = VidLoader(train_dataset, clip_model, vae_model, device, batch_size=batch_size, shuffle=True)
        val_loader = VidLoader(val_dataset, clip_model, vae_model, device, batch_size=batch_size)
        
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step()
    
    # Final test
    val_loss = validate(model, val_loader, criterion, device)
    logging.info(f"Final Test Loss: {val_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "video_prediction_model.pth")
    logging.info("Model saved. Training completed.")

if __name__ == "__main__":
    main()