import pickle
import torch

import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse 
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import markov_clustering as mc
import networkx as nx
import scipy.sparse as sp
from sklearn.cluster import KMeans
def statistics(filter):
    mean_value = np.mean(filter)
    max_value = np.max(filter)

    # Calculate quantiles (25%, 50%, and 75%)
    q25 = np.percentile(filter, 25)
    q50 = np.percentile(filter, 50) 
    q75 = np.percentile(filter, 75)

    # Print the results
    print(f"Mean: {mean_value}")
    print(f"Max: {max_value}")
    print(f"25th Percentile: {q25}")
    print(f"50th Percentile (Median): {q50}")
    print(f"75th Percentile: {q75}")



def mcl(key, value,args,dev_count_file):
    if args.filter_count_json is not None:
        if not  key in dev_count_file:
            return key,  torch.tensor([0])

    if value.shape[0] < args.occurance:
        if args.bf16:
            return  key, torch.tensor(value).to(torch.bfloat16)
        else:
            return  key, torch.tensor(value).to(torch.float16)


    processed_value = value


   
    if args.metric == "l1":
        diffs = squareform(pdist(processed_value, metric='cityblock'))
        diffs = 1 / (diffs + 0.01)
        sparse_matrix = diffs
    elif args.metric == "dense":
        diffs = squareform(pdist(processed_value, metric='euclidean'))
        diffs = 1 / (diffs + 0.01)
        sparse_matrix = diffs
    elif args.metric == "e":
        diffs = squareform(pdist(processed_value, metric='euclidean'))
        diffs = np.exp(-1 * args.distance * diffs)
        sparse_matrix = diffs
    elif args.metric =="kmeans":

        kmeans = KMeans(n_clusters=int(args.distance), random_state=0,n_init="auto").fit(processed_value)
        centers = kmeans.cluster_centers_

        if args.bf16:
            return key, torch.tensor(centers).to(torch.bfloat16)
        else:
            return key, torch.tensor(centers).to(torch.float16)

    if args.elastic < 0:
        result = mc.run_mcl(sparse_matrix, inflation=args.inflation, expansion = args.expansion)           # run MCL with default parameters
        clusters = mc.get_clusters(result)    # get clusters


    else:
        generate_cluster = False
        current_inflation = args.inflation 
        while ( current_inflation > 1.0):
            result = mc.run_mcl(sparse_matrix, inflation=current_inflation, expansion = args.expansion)      
            clusters = mc.get_clusters(result)
            if (len (clusters)) < args.occurance or (len(clusters)/value.shape[0] <=args.elastic ):  
                generate_cluster = True 
                break 
            current_inflation = current_inflation - args.inflation_decrease
        if not generate_cluster:
            print("no generate",key, value.shape,len(clusters))
            result = mc.run_mcl(sparse_matrix, inflation=args.inflation, expansion = args.expansion)          
            clusters = mc.get_clusters(result)
    



    cluster_medoids = list()
    for c_indices in clusters:
        c = value[c_indices,:]
        centroid = np.mean(c, axis=0)
        cluster_medoids.append(centroid)
    try :
        cluster_medoids = np.vstack(cluster_medoids)
    except:
        print(clusters)
    if args.bf16:
        return key, torch.tensor(cluster_medoids).to(torch.bfloat16)
    else:
        return key, torch.tensor(cluster_medoids).to(torch.float16)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, help="output pkl file")
    parser.add_argument('--continue_file', type=str, default='none', help="output pkl file")
    parser.add_argument('--filter_count_json', type=str, default=None, help="output pkl file")
    parser.add_argument('--output_file', type=str, help="output pkl file")
    parser.add_argument('--distance', type=float, default=10, help="output pkl file") # not use
    parser.add_argument('--expansion', type=int, default=2, help="output pkl file")
    parser.add_argument('--inflation', type=float, default=1.75, help="output pkl file")
    parser.add_argument('--data_prepossessing', type=str, default="none", help="output pkl file")
    parser.add_argument('--metric', type=str, default="kmeans", help="output pkl file")
    parser.add_argument('--occurance', type=int, default=10, help="output pkl file")  # 20 is also the same
    parser.add_argument('--elastic', type=float, default=0.2, help="output pkl file")  
    parser.add_argument('--inflation_decrease', type=float, default=0.05, help="output pkl file") 
    parser.add_argument('--bf16', type=bool, default=False, help="output pkl file") 
    args = parser.parse_args()
    print(args)
    print("input file:",args.input_file)
    with open(args.input_file, 'rb') as handle: 
        data = pickle.load(handle)

   
    output_data = dict()

    if args.filter_count_json is not None:
        print(f"Load filer count from {args.filter_count_json}")
        with open(args.filter_count_json, 'rb') as handle: 
            dev_count_file = pickle.load(handle)
    else :
        dev_count_file = dict()

    '''
    with open(args.continue_file, 'rb') as handle: 
        output_data = pickle.load(handle)
    '''
    cluster_centroids_data = dict()
    cluster_medoids_data = dict()
    import random
    idx = 0
    keys = list(data.keys())
    random.shuffle(keys)

    for key in tqdm(keys):

        value = data[key]
        output_key, output_value = mcl(key,value,args,dev_count_file)
        output_data[output_key] = output_value


 
    print("save to:",args.output_file)
    with open(args.output_file, 'wb') as handle:
        pickle.dump(output_data, handle)
    



