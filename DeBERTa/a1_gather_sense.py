import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
import argparse
import random
import pickle
from tqdm import tqdm
import numpy as np
import os
import time
import multiprocessing
import json
from collections import OrderedDict 
import torch.nn.functional as F
from a4_deberta_models import TeacherModel, StudentModel
from DeBERTa.deberta.spm_tokenizer import SPMTokenizer



def _truncate_segments(segments, max_num_tokens=128):
  """
  Truncate sequence pair according to original BERT implementation:
  https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
  """
  while True:
    if sum(len(s) for s in segments)<=max_num_tokens:
      break

    segments = sorted(segments, key=lambda s:len(s), reverse=True)
    trunc_tokens = segments[0]

    assert len(trunc_tokens) >= 1
    trunc_tokens.pop()
  return segments

class StringPairDataset(Dataset):

    def __init__(self, filepath, tokenizer, start_step=0):


        with open(filepath, 'r', encoding='utf-8') as file:
            self.pairs  = json.load(file)
        self.tokenizer = tokenizer
        self.start_step = start_step

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        max_seq_len = 128
        prompts = self.pairs[idx]
        segments = _truncate_segments([self.tokenizer.tokenize(s) for s in prompts], 128)
        tokens = ['[CLS]']
        type_ids = [0]
        for i,s in enumerate(segments):
            tokens.extend(s)
            tokens.append('[SEP]')
            type_ids.extend([i]*(len(s)+1))

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        pos_ids = list(range(len(token_ids)))
        input_mask = [1]*len(token_ids)
        features = OrderedDict(input_ids = token_ids,
            type_ids = type_ids,
            position_ids = pos_ids,
            input_mask = input_mask)

        return features


def pad_tensors(lists, pad_value=0):

    max_length = max(len(sublist) for sublist in lists)
    
    # Pad each sublist to the maximum length
    padded_lists = [sublist + [pad_value] * (max_length - len(sublist)) for sublist in lists]
    padded_lists = torch.tensor(padded_lists)
    return padded_lists
def collate_fn(batch):
    input_ids = []
    type_ids = []
    position_ids = []
    input_mask = []
    for item in batch:
        if isinstance(item, dict):
            input_ids.append(item['input_ids'])
            type_ids.append(item['type_ids'])
            position_ids.append(item['position_ids'])
            input_mask.append(item['input_mask'])
        else:
            continue
 
    # Pad the input_ids and attention_masks
    input_ids_padded = pad_tensors(input_ids)
    type_ids_padded = pad_tensors(type_ids)
    position_ids_padded = pad_tensors(position_ids)
    input_mask_padded = pad_tensors(input_mask)


    return {
        'input_ids': input_ids_padded,
        'type_ids': type_ids_padded,
        'position_ids' :  position_ids_padded,
        'input_mask' : input_mask_padded
    }
    

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='./gather_json/combine_train.json', help="input json file")
    parser.add_argument('--count_file', type=str, default='./gather_json/combine_train.json', help="input json file")
    parser.add_argument('--output_file', type=str, default='./mrpc_minikmeans_5', help="Output KeyWord")
    parser.add_argument('--teacher_ckpt_path', type=str, default=None, help="Teacher Checkpoint path")
    parser.add_argument('--num_labels', type=int, default=2, help='Number of Labels')
    parser.add_argument('--k', type=int, default=1000, help="output pkl file")


    args = parser.parse_args()

    with open(args.count_file, 'rb') as handle:
        count_data = pickle.load(handle)
 
    print("Main Process Start from Beginning!")
    vocab_size = 128000
    cache_dict = dict()
    # Initialize  the DataLoader
    tokenizer = SPMTokenizer( "./cache/assets/latest/deberta-v3-large/spm.model")
    
    model = TeacherModel.load_model( "deberta-v3-large", "./experiments/glue/config.json", num_labels=args.num_labels, \
      drop_out=0,only_return_hidden=True )
  
    if args.teacher_ckpt_path is not None:
        print("Load teacher ckpt from", args.teacher_ckpt_path)
        bin_file_path = args.teacher_ckpt_path
        model_state_dict = torch.load(bin_file_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)
    
    model = model.half()
    model = model.to("cuda:0")
    model.eval()

    dataset = StringPairDataset(args.json_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn, num_workers=4)

    stop_training = False
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids'].to("cuda:0")
            type_ids = batch['type_ids'].to("cuda:0")  
            input_mask = batch['input_mask'].to("cuda:0")
            position_ids = batch['position_ids'].to("cuda:0")


            outputs = model(input_ids=input_ids, type_ids=type_ids, input_mask=input_mask,  position_ids=position_ids)
            embeddings = outputs.detach().cpu().numpy()

            for i in range(embeddings.shape[0]):
                for j in range(embeddings.shape[1]):
                    token_id = input_ids[i, j].item()
                    if input_mask[i][j]  == 0:
                            continue
                    cache_dict.setdefault(token_id, list())
                    try:
                        count = count_data[token_id]
                    except:
                        count = 0
                        print(f"id {token_id} is missing in count_data")
                    if count > args.k and len(cache_dict[token_id]) > args.k:
                        random_integer = random.randint(1, count_data[token_id])
                        if random_integer < args.k:
                            cache_dict[token_id].pop(0)
                            cache_dict[token_id].append(embeddings[i,j])
                    else:
                        cache_dict[token_id].append(embeddings[i,j])
    output_dict = dict()
    for key, value in cache_dict.items():
        output_dict[key] = np.vstack(value)

    print("save to",args.output_file)
    with open(args.output_file, 'wb') as handle:
        pickle.dump(output_dict, handle)


