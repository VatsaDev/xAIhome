import csv
import os
import subprocess
from collections import defaultdict
from pytubefix import YouTube

def download_youtube_video(url, output_path):
    try:
        yt = YouTube(url)
        video = yt.streams.get_highest_resolution()
        video.download(output_path=output_path)
        return True
    except Exception as e:
        print(f"Failed to download video {url}: {type(e).__name__}: {str(e)}")
        return False

def load_labels(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return [line.strip() for line in f]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the attempted encodings: {encodings}")

label_counts = defaultdict(int)
video_data = defaultdict(list)

try:
    labels = load_labels('labels.txt')
except ValueError as e:
    print(f"Error loading labels: {e}")
    exit(1)

with open('train.txt', 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file, 1):
        line = line.strip()
        try:
            # Split the line into URL and label part
            url_part, label_part = line.rsplit(' ', 1)
            
            # Extract the first label index (before the comma, if present)
            label_index = int(label_part.split(',')[0])
            
            if label_index < 0 or label_index >= len(labels):
                raise ValueError(f"Invalid label index: {label_index}")
            
            label = labels[label_index]
            url = url_part.strip()
            
            youtube_id = url.split('=')[-1]
            label_counts[label] += 1
            video_data[youtube_id].append((label, url))
            print(f"Successfully processed line {line_number}: {url} - Label: {label}")  # New success print
        except Exception as e:
            print(f"Error processing line {line_number}: {line}")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            continue

# Process videos
for youtube_id, clips in video_data.items():
    for label, url in clips:
        os.makedirs(f"vids/{label}", exist_ok=True)
        output_file = f"vids/{label}/{youtube_id}.mp4"
        
        if not os.path.exists(output_file):
            success = download_youtube_video(url, f"vids/{label}")
            
            if success:
                # Rename the downloaded file to match our desired format
                downloaded_file = max([f for f in os.listdir(f"vids/{label}") if f.endswith('.mp4')], key=lambda x: os.path.getctime(os.path.join(f"vids/{label}", x)))
                os.rename(os.path.join(f"vids/{label}", downloaded_file), output_file)
                
                print(f"Successfully downloaded: {output_file}")
            else:
                print(f"Failed to download: {url}")

print("Label counts:")
print(label_counts)
lowest_count = min(label_counts.values()) if label_counts else 0
print(f"Lowest count: {lowest_count}")