import multiprocessing
import os
import glob
import shutil
import argparse
import time
from collections import defaultdict
from multiprocessing import Pool
import psutil


from aos_wrapper import generate_integral


# To use it:
start_time = time.time()

def print_timestamp(description):
    # Get the current time

    if start_time is not None:
        current_time = time.time()
        elapsed_time = current_time - start_time
    else:
        elapsed_time = 0

    # Print the timestamp
    print(description+" : {:.2f}s".format(elapsed_time))

    return elapsed_time


# Do some operations...

def process_file(args):
    sample_id, files, id_replace_map, dst_dir = args
    new_sample_id = id_replace_map.get(sample_id)
    if new_sample_id is not None:
        for filepath in files:
            try:
                new_name = filepath.replace(f"_{sample_id}_", f"_{new_sample_id}_")
                dest_path = os.path.join(dst_dir, os.path.basename(new_name))
                shutil.copy(filepath, dest_path)
            except OSError as e:
                print(f"Got OSError: {e}, trying again for file: {filepath}")
                time.sleep(5)
                new_name = filepath.replace(f"_{sample_id}_", f"_{new_sample_id}_")
                dest_path = os.path.join(dst_dir, os.path.basename(new_name))
                shutil.copy(filepath, dest_path)


def process_images(num_processes, batch_id, file_list, dst_dir):
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

    print(f"Excluded {excluded_count} samples from training.")
    # The iterable we will be parallel-processing.
    iterable = [(sample_id, files, id_replace_map, dst_dir) for sample_id, files in file_list.items()]

    # Create a pool of processes, and map the function to the iterable
    with Pool() as p:
        p.map(process_file, iterable)

    focal_planes = (0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.6, -2.0, -2.4)
    pre_integral = print_timestamp("Files created.")
    print(f"Generating {len(id_replace_map)} samples with {len(focal_planes)} focal planes each...")

    def split_into_sublists(my_list, n):
        sublist = []
        size_of_each_part = len(my_list) // n
        remainder = len(my_list) % n
        iterator = iter(my_list)

        for i in range(n):
            sublist_length = size_of_each_part + (i < remainder)
            sublist.append(list(next(iterator) for _ in range(sublist_length)))

        return sublist

    chunked_tasks = list(split_into_sublists(list(id_replace_map.values()), num_processes))

    chunked_tasks = [([(batch_id, new_sample_id) for new_sample_id in chunk], focal_planes, None) for chunk in chunked_tasks]

    chunked_tasks = [c for c in chunked_tasks if len(c[0])]

    print(f"Launching up to {num_processes} processes.")

    # Create a pool of processes, and map the function to the iterable
    with Pool(processes=num_processes) as p:
        p.map(generate_integral, chunked_tasks)

    post_integral = print_timestamp(f"Integrals generated.")

    print("Time taken per focal plane: {:.3f}s".format((post_integral - pre_integral)/(len(id_replace_map) * len(focal_planes))))

    print("Finished.")


def main():
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

    parser.add_argument(
        "--processes",
        type=int,
        required=False,
        default=psutil.cpu_count(logical=False) - 1,
        help="Number of worker processes to start"
    )
    # Parse arguments
    args = parser.parse_args()

    # Get the current process
    process = psutil.Process(os.getpid())

    # Set the priority of the process. Note that 'high' priority corresponds
    # to 'ABOVE_NORMAL_PRIORITY_CLASS' in windows, not 'HIGH_PRIORITY_CLASS'.
    process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)

    # Map batches to identifier
    batch_id_map = {"batch_20230912": 0, "batch_20230919": 1, "batch_20231027": 2}

    dst_dir = "./data/train"

    # Check if directory exists and has files
    if os.path.exists(dst_dir) and os.listdir(dst_dir):
        proceed = input(f"Clear {dst_dir} of existing files? (y/n): ")
        if proceed.lower() == "y":
            print_timestamp("Deleting files. ")

            files = glob.glob("./data/train/*")
            for f in files:
                os.remove(f)
            print_timestamp(f"Deleted {len(files)} files.")

    os.makedirs(dst_dir, exist_ok=True)

    if args.limit is not None:
        print(f"Limiting output to {args.limit} samples.")

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

    print_timestamp("Processing...")
    process_images(args.processes, batch_id, batch_file_list, dst_dir)


if __name__ == "__main__":
    main()
