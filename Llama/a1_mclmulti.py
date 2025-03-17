import torch
import numpy as np
from tqdm import tqdm
import argparse 
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import markov_clustering as mc
import pickle

#  MCL function to process each item
def mcl(item, args):
    key, value = item
    # Skip clustering if the number of occurrences is less than the threshold
    if value.shape[0] < args.occurance:
        return key, torch.tensor(value).to(torch.bfloat16)

    processed_value = value 
    if args.metric == "l1":  # L1 or Manhattan distance
        diffs = squareform(pdist(processed_value, metric='cityblock'))
        diffs = 1 / (diffs + 0.01) 
        sparse_matrix = diffs
    elif args.metric == "dense":  # Euclidean distance
        diffs = squareform(pdist(processed_value, metric='euclidean'))
        diffs = 1 / (diffs + 0.01)
        sparse_matrix = diffs

    # MCL
    result = mc.run_mcl(sparse_matrix, inflation=args.inflation, expansion=args.expansion)
    clusters = mc.get_clusters(result)  

    # Calculate cluster centroids
    cluster_centroids = list()
    for c_indices in clusters:
        c = value[c_indices, :]  
        centroid = np.mean(c, axis=0)  #
        cluster_centroids.append(centroid)

    # Stack centroids into a single PyTorch tensor
    cluster_centroids = np.vstack(cluster_centroids)
    return key, torch.tensor(cluster_centroids).to(torch.bfloat16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, 
                        help="Path to the input .combinepkl file containing at most 1000 encodings per token.")
    parser.add_argument('--output_path', type=str,
                        help="Path to save the output .mclpkl file containing mcl centroids per token")
    parser.add_argument('--expansion', type=int,
                        help="Expansion parameter for the MCL algorithm, controlling the flow propagation.")
    parser.add_argument('--inflation', type=float, 
                        help="Inflation parameter for the MCL algorithm, controlling the granularity of clusters.")
    parser.add_argument('--metric', type=str, default="dense",
                        help="Clustering metric to use. Options include 'dense', 'l1.")
    parser.add_argument('--occurance', type=int,
                        help="Minimum number of occurrences for tokens to be clustered.")
    args = parser.parse_args()

    print(args)
    print("Input file:", args.input_path)
    with open(args.input_path, 'rb') as handle:
        data = pickle.load(handle)

    output_data = dict()

    # Number of threads for parallel processing
    num_threads = 8
    data_with_args = partial(mcl, args=args) 

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Apply mcl to each item in the dictionary
        output_data = dict(tqdm(executor.map(data_with_args, data.items()),
                                total=len(data),
                                desc="Processing"))

    
    output_path = args.output_path
    print("Save to:", output_path)
    with open(output_path, 'wb') as handle:
        pickle.dump(output_data, handle)
