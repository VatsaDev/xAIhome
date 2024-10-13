import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import time
import subprocess
import tempfile
import shutil

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
        print(f"Error processing video {video}: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error processing video {video}: {str(e)}")
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
            print(f"Error creating crop {crop_index} for {label}_pair_{pair_index}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error creating crop {crop_index} for {label}_pair_{pair_index}: {str(e)}")
    
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

def main():
    start_time = time.time()
    print("Starting video clip creation...")
    root_dir = "vids"
    output_dir = "clips5"
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