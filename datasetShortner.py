import os
import shutil
import random
import json

# Define paths for original and new datasets
original_data_path = "./data"
new_data_path = "./data_small"
annotation_file = os.path.join(original_data_path, "annotation_file", "R3_all.jsonl")
new_annotation_file = os.path.join(new_data_path, "annotation_file", "R3_all.jsonl")

# Create the directory structure for the new dataset
os.makedirs(os.path.join(new_data_path, "annotation_file"), exist_ok=True)
os.makedirs(os.path.join(new_data_path, "raw_videos"), exist_ok=True)

# Set the percentage of rows to sample
sample_percentage = 20  # Adjust the percentage as needed

# Read and sample the annotation file
with open(annotation_file, "r") as f:
    # Read all rows
    lines = f.readlines()

# Extract header and data rows
header = lines[0]
data = lines[1:]

# Calculate the number of samples based on percentage
num_samples = max(1, int((sample_percentage / 100) * len(data)))

# Sample data rows
sampled_data = random.sample(data, num_samples)

# Copy the sampled annotations to the new file
with open(new_annotation_file, "w") as f:
    f.write(header)  # Write the header
    f.writelines(sampled_data)  # Write the sampled rows

# Copy the corresponding video files
raw_videos_path = os.path.join(original_data_path, "raw_videos")
new_raw_videos_path = os.path.join(new_data_path, "raw_videos")

for row in sampled_data:
    try:
        # Parse the JSONL row
        # Assuming each line is a JSON object (not a list)
        row_data = json.loads(row.strip())

        # Check if row_data has the expected fields
        if not isinstance(row_data, list) or len(row_data) <= 2:
            print(f"Skipping malformed row: {row.strip()}")
            continue

        video_filename = row_data[2]  # "vid_filename" is the third column

        # Copy the video file
        src_video_path = os.path.join(raw_videos_path, video_filename)
        dst_video_path = os.path.join(new_raw_videos_path, video_filename)

        if os.path.exists(src_video_path):
            shutil.copy2(src_video_path, dst_video_path)
        else:
            print(f"Warning: Video file {video_filename} not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {row.strip()}. Error: {e}")
    except IndexError as e:
        print(f"IndexError with row: {row.strip()}. Error: {e}")
    except Exception as e:
        print(f"Unexpected error with row: {row.strip()}. Error: {e}")
