
echo "----------------------------------------------------"
echo "Run MCL on the combinepkl data"
echo "----------------------------------------------------"


python a1_mclmulti.py \
--input_path ./sense_dict/combine_1000.combinepkl \
--output_path ./sense_dict/mcl_cluster.mclpkl \
--inflation 1.65 \
--expansion 2 \
--occurance 5 

    



echo "----------------------------------------------------"
echo "Kmeans on the combinepkl data"
echo "----------------------------------------------------"


python a1_kmeans_withmcl.py \
--input_path ./sense_dict/combine_1000.combinepkl \
--input_mclpath ./sense_dict/mcl_cluster.mclpkl \
--output_path ./sense_dict/kmeans_cluster.kmeanspkl \
--k_percent_below 0.1 \
--k_threshold 900 \
--k_percent_above 0.4 