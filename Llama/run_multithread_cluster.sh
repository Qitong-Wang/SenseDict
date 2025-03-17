export PYTHONPATH=$(pwd)

echo "----------------------------------------------------"
echo "Run MCL on the combinepkl data"
echo "----------------------------------------------------"

# Array of PIDs
PIDS=()
for i in {0..5}; do  # 6 processes
    python a1_mclmulti.py \
        --input_path ./sense_dict/combine_1000.part${i}combinepkl \
        --output_path ./sense_dict/mcl_cluster.part${i}mclpkl \
        --inflation 1.65 \
        --expansion 2 \
        --occurance 5 &

    # Save the PID
    PIDS+=($!)
done
# Wait for all background processes
wait "${PIDS[@]}"


echo "----------------------------------------------------"
echo "Kmeans on the combinepkl data"
echo "----------------------------------------------------"

PIDS=()
for i in {0..5}; do
    python a1_kmeans_withmcl.py \
        --input_path ./sense_dict/combine_1000.part${i}combinepkl \
        --input_mclpath ./sense_dict/mcl_cluster.part${i}mclpkl \
        --output_path ./sense_dict/kmeans_cluster.part${i}kmeanspkl \
        --k_percent_below 0.1 \
        --k_threshold 900 \
        --k_percent_above 0.4 &

    # Save the PID
    PIDS+=($!)
done
# Wait for all background processes
wait "${PIDS[@]}"


echo "----------------------------------------------------"
echo "Combine the Kmeans clusters to a single sense dictionary file"
echo "----------------------------------------------------"

PATHS=""
for i in {0..5}; do
    PATHS+=" ./sense_dict/kmeans_cluster.part${i}kmeanspkl"
done
python a1_combine_pkl.py \
    --output_path ./sense_dict/kmeans_cluster.kmeanspkl \
    --paths $PATHS

python s1_count_singlefile.py --input_path ./sense_dict/kmeans_cluster.kmeanspkl


