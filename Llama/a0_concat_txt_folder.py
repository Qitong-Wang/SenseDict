import os
import json
import argparse
def load_json_files_from_folder(folder_path):
    combined_list = []
    # Get all JSON files in the folder and sort them by file name
    json_files = sorted([file_name for file_name in os.listdir(folder_path) if file_name.endswith('.partjson')])
    print(json_files)
    for file_name in json_files:
        # Check if the file ends with '.json'
        if file_name.endswith('.partjson'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                # Ensure that the content is a list of strings
                if isinstance(data, list) and all(isinstance(item, str) for item in data):
                    combined_list.extend(data)
                else:
                    print(f"File {file_name} does not contain a list of strings.")

    return combined_list

def save_combined_list_to_json(combined_list, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(combined_list, json_file, indent=4)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=str, required=True, help="The path to the folder containing the input files.")
parser.add_argument("--output_file_path", type=str, required=True, help="The file path where the output will be saved.")
args = parser.parse_args()

combined_list = load_json_files_from_folder(args.folder_path)
save_combined_list_to_json(combined_list, args.output_file_path)

print(f"Combined list saved to {args.output_file_path}")
