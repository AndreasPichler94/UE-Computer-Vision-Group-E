import os
import shutil
from collections import defaultdict

base_dir = "./data/raw/batch_20230912_part1/Part1"  # for example, adjust as needed
dst_dir = "./data/train"

os.makedirs(dst_dir, exist_ok=True)


def process_images(file_list, dst_dir):
    id_replace_map = {}
    unused_id = 0
    for sample_id in sorted(file_list.keys()):
        if len(file_list[sample_id]) == 12:
            id_replace_map[sample_id] = unused_id
            unused_id += 1
    for sample_id, files in file_list.items():
        new_sample_id = id_replace_map.get(sample_id)
        if new_sample_id is not None:
            for filepath in files:
                new_name = filepath.replace(f"_{sample_id}_", f"_{new_sample_id}_")
                dest_path = os.path.join(dst_dir, os.path.basename(new_name))
                shutil.copy(filepath, dest_path)


batch_file_list = defaultdict(list)
batch_id = "0"  # as per your assumption, it's constant
for dir_path, sub_dir_list, file_list in os.walk(base_dir):
    for file_name in file_list:
        if file_name.startswith(f"{batch_id}_") and file_name.endswith(".png"):
            sample_id = file_name.split("_")[1]
            batch_file_list[sample_id].append(os.path.join(dir_path, file_name))

print("Processing...")
process_images(batch_file_list, dst_dir)
