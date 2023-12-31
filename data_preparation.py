import os
import math
import re
import shutil

last_valid_sequence = 0

def split_data(train_percentage, part_folder):
    print("Splitting data...")

    if not os.path.exists(training_data_path):
        os.makedirs(training_data_path)

    if not os.path.exists(validation_data_path):
        os.makedirs(validation_data_path)

    prefixes = set()
    for filename in os.listdir(os.path.join(raw_directory_path, part_folder)):
        prefix = re.match(r'^(\d+)_', filename)
        if prefix:
            prefixes.add(prefix.group(1))

    for prefix in prefixes:

        file_count = len([f for f in os.listdir(os.path.join(raw_directory_path, part_folder))
                          if f.startswith(f'{prefix}_') and os.path.isfile(os.path.join(raw_directory_path, part_folder, f))])

        total_sequences = round(file_count / 13) #TODO: Wenn hier eine ungerade Zahl herauskommt, wurden nicht alle Files richtig sortiert.
        train_size = round((file_count / 13) * train_percentage)

        # Copy files according to the split into training and validation data directories.
        for current_index in range(1, total_sequences):
            expected_files = get_expected_files(current_index, part_folder, prefix)

            if current_index < train_size:
                destination_path = training_data_path
            else:
                destination_path = validation_data_path

            for expected_file in expected_files:
                source_path = os.path.join(raw_directory_path, expected_file)
                destination_file = os.path.join(destination_path, os.path.basename(expected_file))

                # Check if the file exists before attempting to move it
                if os.path.exists(source_path):
                    shutil.move(source_path, destination_file)
                #else:
                    #print(f"File not found: {source_path}")

    if not last_valid_sequence == 0:
        print(f"Data split into {train_percentage * 100}% training ({train_size} Sequences) and {100 - train_percentage * 100}% validation ({total_sequences - train_size} Sequences).")

    print(f"{os.path.join(raw_directory_path, part_folder)} Splitting complete!")

def get_expected_files(current_index, part_folder, prefix):
    expected_files = set()

    for i in range(11):
        expected_files.add(f'{prefix}_{current_index}_pose_{i}_thermal.png')

    expected_files.add(f'{prefix}_{current_index}_Parameters.txt')
    expected_files.add(f'{prefix}_{current_index}_GT_pose_0_thermal.png')

    return [os.path.join(part_folder, file) for file in expected_files]

def rename_files(current_index, part_folder, prefix):
    expected_files = get_expected_files(current_index, part_folder, prefix)
    for expected_file in expected_files:
        old_path = os.path.join(raw_directory, expected_file)
        new_index = last_valid_sequence + 1
        new_file = expected_file.replace(f'{prefix}_{current_index}', f'{prefix}_{new_index}')
        new_path = os.path.join(raw_directory, new_file)
        os.rename(old_path, new_path)
    #print(f"Renamed sequences with index {current_index} to {last_valid_sequence + 1}")
    return new_index

def cleanup_data(part_folder):
    print(f"{raw_directory} - Cleaning started...")

    global last_valid_sequence
    invalid_sequences_found = False

    prefixes = set()
    for filename in os.listdir(os.path.join(raw_directory_path, part_folder)):
        prefix = re.match(r'^(\d+)_', filename)
        if prefix:
            prefixes.add(prefix.group(1))

    for prefix in prefixes:
        last_valid_sequence = 0

        file_count = len([f for f in os.listdir(os.path.join(raw_directory_path, part_folder))
                          if f.startswith(f'{prefix}_') and os.path.isfile(os.path.join(raw_directory_path, part_folder, f))])

        # TODO better calculation for sequences where more than 50% of the Files are missing.
        approximated_sequences = math.ceil(file_count / 13)

        for current_index in range(approximated_sequences):
            expected_files = get_expected_files(current_index, part_folder, prefix)
            missing_file = None
            found_files = [file_name for file_name in expected_files if
                           os.path.exists(os.path.join(raw_directory_path, part_folder, file_name))]

            if len(found_files) > 0:
                for file_name in expected_files:
                    full_path = os.path.join(raw_directory_path, part_folder, file_name)
                    if not os.path.exists(full_path):
                        missing_file = file_name
                        break

                if missing_file is not None:
                    invalid_sequences_found = True
                    for delete_file in expected_files:
                        delete_path = os.path.join(raw_directory_path, part_folder, delete_file)
                        if os.path.exists(delete_path):
                            os.remove(delete_path)
                    print(f"Sequence {current_index} (Prefix {prefix}) deleted due to missing file: {missing_file}")
                else:
                    if current_index != 0 and (current_index - last_valid_sequence) > 1:
                        last_valid_sequence = rename_files(current_index, part_folder, prefix)
                    else:
                        last_valid_sequence = current_index

    if not invalid_sequences_found:
        print("No invalid sequences found.")


if __name__ == "__main__":
    #TODO Unzip all the zipped batches into ./raw/data
    raw_directory_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(__file__), '..')), 'data', 'raw')
    training_data_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(__file__), '..')), 'data', 'train')
    validation_data_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(__file__), '..')), 'data', 'test')
    train_percentage = 0.8

    # Process Folders
    for part_folder in ["Part1", "Part2"]:
        raw_directory = os.path.join(raw_directory_path, part_folder)

        # Data processing
        cleanup_data(raw_directory)
        split_data(train_percentage, raw_directory) #TODO: Improving.
        print("##########################################")
