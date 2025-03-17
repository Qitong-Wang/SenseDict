
'''
print( '/gpfs/u/scratch/DLTM/DLTMwngq/decoder/cls-mteb/llm2vec/llm2vec.py import mteb_eval')

import argparse
import mteb
import json
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pickle
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# FOR DCS ONLY
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
)
parser.add_argument("--task_name", type=str, default="STS16")
parser.add_argument(
    "--task_to_instructions_fp",
    type=str,
    default="test_configs/mteb/task_to_instructions.json",
)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument('--sense_func', type=str, default='dot', help="output pkl file")
parser.add_argument('--sense_dict_path', type=str, default=None, help="output pkl file")
parser.add_argument('--freq', type=int, default=-1, help="output pkl file")
parser.add_argument('--run_option', type=str, default="replace", help="output pkl file") #replace, cluster, origin, save_txt
parser.add_argument('--save_path', type=str, default="./STS_encodings/llama3_STS16_fp32_.pkl", help="output pkl file")
parser.add_argument('--d', type=int, default=2, help="output pkl file")
args = parser.parse_args()

import threading
save_path=args.save_path
if args.sense_dict_path is not None:
    with open(args.sense_dict_path, 'rb') as handle:
        sense_dict = pickle.load(handle)

#with open("./STS_encoding_test/STS_freq.mclpkl", 'rb') as handle:
#    omit_dict = pickle.load(handle)

#for key, value in omit_dict.items():
#    sense_dict[key] = value

#print("Run with OMIT!")

sense_func = args.sense_func
run_option = args.run_option
d = args.d
def main(args):
    print(args.run_option)
    model_kwargs = {}
    if (args.task_to_instructions_fp is not None) and (not args.task_to_instructions_fp == "None") :
        with open(args.task_to_instructions_fp, "r") as f:
            task_to_instructions = json.load(f)
        model_kwargs["task_to_instructions"] = task_to_instructions

    model = mteb.get_model(args.model_name, **model_kwargs)
    tasks = mteb.get_tasks(tasks=[args.task_name])
    evaluation = mteb.MTEB(tasks=tasks)

    results = evaluation.run(model, output_folder=args.output_dir)


if __name__ == "__main__":

    main(args)
    #if args.sense_dict_path is not None:
    #    with open(args.sense_dict_path, 'rb') as handle:
    #        sense_dict = pickle.load(handle)
    #sense_func = args.sense_func

    # [5347, 264, 315, 1174, 311, 304, 662, 128009, 374, 323, 364, 1148, 433, 602, 656, 499, 949]
    #[sem   a     of   ,    to    in  .     <eot>   is   and  '    what  it  i    do    you   ?]
 

    #base_list = [ 264, 1174, 304, 662, 128009, 374, 323, 364, 1148, 433, 602, 656, 499, 949]
    #base_list.pop(args.freq)
    #print(base_list)
    #freq_id =base_list
    


'''