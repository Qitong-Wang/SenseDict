import pickle
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    output_data = dict()

    parser = argparse.ArgumentParser(description="Combine multiple .pkl files into a single dictionary.")
    parser.add_argument("--output_path", type=str,
                        help="Path to output the .pkl file.")
    parser.add_argument('--paths', nargs='+', help='List of paths to input .pkl files.', required=True)
    args = parser.parse_args()

    for item in args.paths:
        print(f"Processing file: {item}")

    for file_path in tqdm(args.paths, desc="Merging files"):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)  

        for key, value in data.items():
            output_data[key] = value  

    print("Saving combined data to", args.output_path)
    with open(args.output_path, 'wb') as handle:
        pickle.dump(output_data, handle)
