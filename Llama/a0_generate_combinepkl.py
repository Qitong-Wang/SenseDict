import pickle
import argparse
import os
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_names', type=str, nargs='+', default=['./CLS_encoding_train/'],
                        help="List of folder names to search for pkl files")
    parser.add_argument('--output_keyword', type=str, default='./CLS_encoding_train/CLS',
                        help="Output pkl file keyword")
   
    args = parser.parse_args()

    cache_dict = {}

    # Iterate over each folder provided in the arguments
    for folder in args.folder_names:
        for root, dirs, files in os.walk(folder):
            for filename in files:
                # Check if the file is a .pkl file
                if filename.endswith('.pkl'):
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'rb') as file:
                        data = pickle.load(file)
                        print(f"Loaded data from {file_path}")     
                    # Update cache_dict with the contents of data
                    for key, value in data.items():
                        cache_dict.setdefault(key, [])
                        cache_dict[key].extend(value)

    for key, value in cache_dict.items():
        if not (isinstance(value, list) and all(isinstance(x, np.ndarray) for x in value)):
            # If not a numpy array, convert each element to a numpy array.
            value = [np.array(x) for x in value]

        if value.shape[0] > 4000:
            indices = np.random.permutation(value.shape[0])[:4000]
            cache_dict[key] = value[indices]



    # Save the combined data to an output file
    output_path = str(args.output_keyword) + ".combinepkl"
    print("Saving combined data to", output_path)
    with open(output_path, 'wb') as handle:
        pickle.dump(cache_dict, handle)
