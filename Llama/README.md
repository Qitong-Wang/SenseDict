# Knowledge Distillation on LLM2Vec Llama3-8b-Instruct with MTEB Classification Tasks

## Overview

This repository provides an implementation of knowledge distillation using the **LLM2Vec Llama3-8b-Instruct** model, specifically applied to **MTEB** classification tasks. Our approach builds upon the **LLM2Vec** and **MTEB** repositories, which you can explore here:

- [LLM2Vec Repository](https://github.com/McGill-NLP/llm2vec)  
- [MTEB Repository](https://github.com/embeddings-benchmark/mteb)  

## Step-by-Step Pipeline

This repository provides a streamlined pipeline for extracting sense embeddings from task-specific training datasets and using them for knowledge distillation. Here is the the process for single substasks, which is lightweight and easy to follow.

1. **Gather sense embeddings**  
   Extract sense embeddings from a given task’s training dataset. It requires a task index (1–16) as an argument.  
   ```bash
   sh run_gather_sense.sh
   ```

1. **Cluster sense embeddings**  
   Generate the **sense dictionary** by clustering the extracted embeddings.  
   ```bash
   sh run_sep_cluster.sh
   ```

2. **Train the student knowledge distillation model**  
   - This script requires a task index (1–16) as an argument.  
   - Additional parameters such as `master_addr`, `master_port`, and `node_rank` may need to be adjusted based on your machine's setup.  
   ```bash
   sh run_train.sh
   ```

### Training on All Datasets for Better Generalization

While training on a **single dataset** works, it is recommended to **train on all datasets** to improve generalization and reduce bias. You can do this by:

1. Running the sense embedding extraction for **each dataset**:  
   ```bash
   sh run_gather_sense.sh
   ```
2. Merging all extracted sense dictionaries:  
   ```bash
   python a0_generate_combinepkl.py --folder_names ./sense_dict/ --output_keyword ./sense_dict/combine_1000
   ```
3. Accelerating clustering using **multithreading**:  
   ```bash
   sh run_multithread_cluster.sh
   ```
4. Adjusting the training configuration to use the combined dataset:  
   - Modify `--json_path` to `./text_train/train.json` to train the concatenated dataset:  
     ```bash
     sh run_train.sh
     ```


### Machine-Specific Optimizations

Due to hardware limitations, certain advanced features, such as fast attention mechanisms, have been disabled. If you have access to a more powerful machine, consider enabling these optimizations for improved performance. Future updates will include more standardized implementations*and better hardware support.

## Dataset and Checkpoints

We provide access to the datasets, sense dictionaries, and model checkpoints used in our experiments. Specifically, we share the **SKD results** from Table 4 in our paper.

**Download Resources**: [Google Drive](https://drive.google.com/file/d/1uI6HbkksVLxuOqsx0uB7eeMLsFeQ57cw/view?usp=sharing)
