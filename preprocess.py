import os
import glob
import shutil
import argparse
from collections import defaultdict

from aos_wrapper import generate_integral

# Create ArgumentParser instance
parser = argparse.ArgumentParser()
# Add argument for batch
parser.add_argument(
    "--batch",
    type=str,
    choices=["batch_20230912", "batch_20230919", "batch_20231027"],
    required=True,
    help="Batch to be processed",
)
# Add optional limit for debugging
parser.add_argument(
    "--limit",
    type=int,
    required=False,
    default=None,
    help="Optional limit for debugging",
)
# Parse arguments
args = parser.parse_args()

# Map batches to identifier
batch_id_map = {"batch_20230912": 0, "batch_20230919": 1, "batch_20231027": 2}

dst_dir = "./data/train"

# Check if directory exists and has files
if os.path.exists(dst_dir) and os.listdir(dst_dir):
    proceed = input(f"Clear {dst_dir} of existing files? (y/n): ")
    if proceed.lower() == "y":
        files = glob.glob("./data/train/*")
        for f in files:
            os.remove(f)
        print(f"Deleted {len(files)} files.")


os.makedirs(dst_dir, exist_ok=True)

if args.limit is not None:
    print(f"Limiting output to {args.limit} samples.")


def process_images(batch_id, file_list, dst_dir):
    id_replace_map = {}
    unused_id = 0
    excluded_count = 0
    for sample_id in sorted(file_list.keys()):
        num_files_present = len(file_list[sample_id])
        if num_files_present == 13:
            id_replace_map[sample_id] = unused_id
            unused_id += 1
        else:
            print(
                f"Got corrupted sample (b: {batch_id} s: {sample_id}) with only {num_files_present} files, excluding."
            )
            excluded_count += 1

    for sample_id, files in file_list.items():
        new_sample_id = id_replace_map.get(sample_id)
        if new_sample_id is not None:
            for filepath in files:
                new_name = filepath.replace(f"_{sample_id}_", f"_{new_sample_id}_")
                dest_path = os.path.join(dst_dir, os.path.basename(new_name))
                shutil.copy(filepath, dest_path)

    generate_integral(
        [(batch_id, new_sample_id) for new_sample_id in id_replace_map.values()],
        focal_planes=(0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.6, -2.0, -2.4),
    )

    print(f"Excluded {excluded_count} samples from training.")


batch_id = batch_id_map[args.batch]
batch_file_list = defaultdict(list)

# Generate pattern for glob with provided batch
pattern = f"./data/raw/{args.batch}*"

# Use glob to find all filenames matching pattern
for foldername in glob.glob(pattern, recursive=True):
    print(f"Scanning items in {foldername}")
    for dir_path, sub_dir_list, file_list in os.walk(foldername):
        for file_name in file_list:
            if file_name.startswith(f"{batch_id}_") and (
                file_name.endswith(".png") or file_name.endswith(".txt")
            ):
                sample_id = file_name.split("_")[1]

                if args.limit is not None and len(batch_file_list) >= args.limit:
                    if sample_id not in batch_file_list.keys():
                        break

                batch_file_list[sample_id].append(os.path.join(dir_path, file_name))


print("Processing...")
process_images(batch_id, batch_file_list, dst_dir)
