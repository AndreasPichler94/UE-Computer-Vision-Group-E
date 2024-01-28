import os
import re

# Define directory
dir_path = "./data/test"

min_batch_index = { b: float('inf') for b in (0, 1, 2)}

# Find the minimum sample index
for filename in os.listdir(dir_path):
    # Extract batch index and sample index from filename using regex
    match = re.match(r"(\d+)_(\d+).+", filename)
    if match:
        batch_index, sample_index = map(int, match.groups())
        min_batch_index[batch_index] = min(min_batch_index[batch_index], sample_index)

# Rename the files, normalizing the sample index
for filename in os.listdir(dir_path):
    # Extract batch index, sample index and rest of filename using regex
    match = re.match(r"(\d+)_(\d+)(.+)", filename)
    if match:
        batch_index, sample_index, rest_filename = match.groups()
        batch_index, sample_index = int(batch_index), int(sample_index)
        # Normalize sample index
        norm_sample_index = int(sample_index) - min_batch_index[batch_index]
        # Create new filename
        new_filename = f"{batch_index}_{norm_sample_index}{rest_filename}"
        # Rename file
        os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, new_filename))