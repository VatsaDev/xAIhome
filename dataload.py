import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import time

def process_video(video, root_dir, output_dir):
    video_path = os.path.join(root_dir, video)
    cap = cv2.VideoCapture(video_path)
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
    
    clips = []
    for pair_index in range(len(frame_pairs) // 2):
        video_pair = frame_pairs[pair_index * 2 : (pair_index * 2) + 2]
        clips.extend(create_crops_from_pair(video_pair, label, pair_index, output_dir))
    
    return clips

def create_crops_from_pair(video_pair, label, pair_index, output_dir):
    clips = []
    for crop_index in range(3):
        crop_x, crop_y = np.random.randint(0, 768-256), np.random.randint(0, 768-256)
        video_crops_1, video_crops_2 = [], []

        for frames in video_pair:
            for frame in frames:
                crop_1 = frame[crop_y:crop_y + 256, crop_x:crop_x + 256]
                crop_2 = frame[crop_y:crop_y + 256, crop_x:crop_x + 256]
                video_crops_1.append(crop_1)
                video_crops_2.append(crop_2)

        clip_name_1 = f"{label}_pair_{pair_index}_crop_{crop_index}_1.npy"
        clip_name_2 = f"{label}_pair_{pair_index}_crop_{crop_index}_2.npy"
        
        save_crop(video_crops_1, os.path.join(output_dir, clip_name_1))
        save_crop(video_crops_2, os.path.join(output_dir, clip_name_2))

        clips.append(clip_name_1)
        clips.append(clip_name_2)

    return clips

def save_crop(frames, filename):
    np.save(filename, np.array(frames))

def main():
    start_time = time.time()
    print("Starting video clip creation...")

    root_dir = "vids"
    output_dir = "clips"
    os.makedirs(output_dir, exist_ok=True)

    videos = [f for f in os.listdir(root_dir) if f.endswith('.mp4')]
    print(f"Found {len(videos)} videos")

    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} workers for multiprocessing")

    with multiprocessing.Pool(num_workers) as pool:
        process_func = partial(process_video, root_dir=root_dir, output_dir=output_dir)
        all_clips = list(tqdm(pool.imap(process_func, videos), total=len(videos), desc="Processing videos"))

    all_clips = [clip for sublist in all_clips for clip in sublist]
    print(f"Total clips created: {len(all_clips)}")

    batch_size = 32
    for i in range(0, len(all_clips), batch_size):
        batch = all_clips[i:i+batch_size]
        print(f"\nBatch {i//batch_size + 1}:")
        for clip in batch:
            print(clip)

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()