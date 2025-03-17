import pickle
import torch
import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        help="Path to the input .combinepkl file containing at most 1000 encodings per token.")
    parser.add_argument('--input_mclpath', type=str,
                        help="Path to the input .mclpkl file containing MCL centroids per token.")
    parser.add_argument('--output_path', type=str, 
                        help="Path to save the output .kmeanspkl file containing kmeans centroids per token.")
    parser.add_argument('--k_percent_below', type=float,
                        help="Percentage multiplier for the number of clusters if MCL cluster size below k_threshold.")
    parser.add_argument('--k_percent_above', type=float,
                        help="Percentage multiplier for the number of clusters if MCL cluster size exceeds k_theshold.")
    parser.add_argument('--k_threshold', type=int, 
                        help="Threshold for deciding whether to use the k_percent_below or k_percent_above multiplier.")
    args = parser.parse_args()

    with open(args.input_path, 'rb') as handle:
        print("Load data from", args.input_path)
        data = pickle.load(handle)
    with open(args.input_mclpath, 'rb') as handle:
        print("Load MCL data from", args.input_mclpath)
        mcl_data = pickle.load(handle)

    # Assert both data and MCL data have the same keys
    assert set(data.keys()) == set(mcl_data.keys()), "Dictionaries do not have the same keys"


    output_data = dict()
    for key, value in tqdm(data.items()):
        # Determine the number of clusters (current_k) 
        if mcl_data[key].shape[0] > args.k_threshold:
            current_k = int(mcl_data[key].shape[0] * args.k_percent_above)
        else:
            current_k = int(mcl_data[key].shape[0] * args.k_percent_below)

        if current_k < 5:
            current_k = 5
        # If the data has fewer rows than the cluster count, skip clustering
        if value.shape[0] <= current_k:
            output_data[key] = torch.tensor(value).to(torch.bfloat16)
        else:
            # K-Means clustering
            numpy_array = value 
            kmeans = KMeans(n_clusters=current_k, random_state=0, n_init="auto").fit(numpy_array)
            cluster_centers = kmeans.cluster_centers_ 
            clustered_tensor = torch.tensor(cluster_centers).to(torch.bfloat16) 
            output_data[key] = clustered_tensor


    with open(args.output_path, 'wb') as handle:
        pickle.dump(output_data, handle)
    print("Output saved to:", args.output_path)
