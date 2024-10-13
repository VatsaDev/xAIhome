import os
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from diffusers import AutoencoderKL
import time
import logging
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VidDataset(Dataset):
    def __init__(self, root_dir, clip_model, vae_model, device):
        self.root_dir = root_dir
        self.clip_model = clip_model
        self.vae_model = vae_model
        self.device = device
        self.clips_dir = "clips"
        os.makedirs(self.clips_dir, exist_ok=True)
        self.data = self.load_data()

    def load_data(self):
        logging.info("Loading data...")
        start_time = time.time()
        
        data = []
        videos = [f for f in os.listdir(self.root_dir) if f.endswith('.mp4')]
        logging.info(f"Found {len(videos)} videos")
        
        for video in tqdm(videos, desc="Processing videos"):
            crops = self.process_video(video)
            data.extend(crops)
        
        end_time = time.time()
        logging.info(f"Total crops processed: {len(data)}")
        logging.info(f"Data loading took {end_time - start_time:.2f} seconds")
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
            crops.extend(self.create_crops_from_pair(video_pair, label, i//2))

        return crops

    def create_crops_from_pair(self, video_pair, label, pair_index):
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

            # Save crops to files
            self.save_crop(vae_f_1, f"pair_{pair_index}_1")
            self.save_crop(vae_f_2, f"pair_{pair_index}_2")

            crops.append((f"pair_{pair_index}_1", clip_f, f"pair_{pair_index}_2"))

        return crops

    def save_crop(self, vae_features, filename):
        np.save(os.path.join(self.clips_dir, f"{filename}.npy"), vae_features.cpu().numpy())

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
        vae_f_1_file, clip_f, vae_f_2_file = self.data[idx]
        vae_f_1 = torch.from_numpy(np.load(os.path.join(self.clips_dir, f"{vae_f_1_file}.npy")))
        vae_f_2 = torch.from_numpy(np.load(os.path.join(self.clips_dir, f"{vae_f_2_file}.npy")))
        return vae_f_1, clip_f, vae_f_2

class VidDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        vae_f_1, clip_f, vae_f_2 = zip(*batch)
        return (
            torch.stack(vae_f_1),
            torch.stack(clip_f),
            torch.stack(vae_f_2)
        )

def main():
    logging.info("Starting main function")
    start_time = time.time()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load CLIP model
    logging.info("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Load VAE model
    logging.info("Loading VAE model...")
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # Create dataset
    logging.info("Creating dataset...")
    dataset = VidDataset("vids", clip_model, vae_model, device)

    # Create dataloader
    logging.info("Creating dataloader...")
    batch_size = 32
    dataloader = VidDataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Process all batches and print clip_f
    logging.info("Processing batches and printing clip_f...")
    for batch_idx, (vae_f_1, clip_f, vae_f_2) in enumerate(tqdm(dataloader, desc="Processing batches")):
        print(f"Batch {batch_idx + 1}, clip_f shape: {clip_f.shape}")
        print(f"clip_f sample:\n{clip_f[0][:10]}")  # Print first 10 elements of the first clip_f in the batch

    end_time = time.time()
    logging.info(f"Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()