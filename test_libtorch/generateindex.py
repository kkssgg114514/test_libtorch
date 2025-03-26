import os
import sys


def generate_feature_paths(root_dir, output_file):
    """
    Recursively search for feature files and generate a path-id mapping file.

    Args:
    - root_dir (str): Root directory to search for feature files
    - output_file (str): Output text file to write paths and ids
    """
    # Dictionary to track unique speakers
    speaker_dict = {}
    current_speaker_id = 1

    # Open output file for writing
    with open(output_file, 'w') as outfile:
        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                # Assuming feature files have a specific extension like .pt or .pth
                if filename.endswith('.feature') or filename.endswith('.pth'):
                    full_path = os.path.abspath(os.path.join(dirpath, filename))

                    # Extract speaker name (could be folder name or part of filename)
                    # Modify this logic based on your actual file/folder naming convention
                    speaker_name = os.path.basename(os.path.dirname(full_path))

                    # Assign or retrieve speaker id
                    if speaker_name not in speaker_dict:
                        speaker_dict[speaker_name] = current_speaker_id
                        current_speaker_id += 1

                    speaker_id = speaker_dict[speaker_name]

                    # Write path and id to output file
                    outfile.write(f"{full_path} {speaker_id}\n")

    print(f"Generated path mapping file: {output_file}")
    print("Speaker mapping:")
    for speaker, id_num in speaker_dict.items():
        print(f"{speaker}: ID {id_num}")


def main():
    generate_feature_paths("M:/fd/testfeature", "M:/fd/filedir.txt")


if __name__ == '__main__':
    main()