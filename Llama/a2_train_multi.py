import argparse
import re
import json
import torch
import os
import torch.distributed
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import json
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pickle
import argparse
import mteb
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import os
from mteb.models.student_models import StudentWrapper
from mteb.models.teacher_models import TeacherWrapper
from transformers import get_scheduler  
import time                           
from transformers import AutoTokenizer 


os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
)
parser.add_argument("--task_name", type=str, default="STS16",help="MTEB task name")
parser.add_argument(
    "--task_to_instructions_fp",
    type=str,
    default="test_configs/mteb/task_to_instructions.json",
)
parser.add_argument("--output_dir", default='./resultdefault/', type=str,help="Directory to store results")
parser.add_argument('--sense_func', type=str, default='dot', help="Functions of finding similarity of sense embeddings")
parser.add_argument('--json_path', type=str, default='./text_train/task_name/train.json', help="Path to the training JSON file")
parser.add_argument('--sense_dict_path', type=str, default="./sense_dict/teacher.kmeanspkl", help="Path to the sense dictionary pickle file")
parser.add_argument('--run_option', type=str, default="train")
parser.add_argument('--save_dir', type=str, help="Directory to save texts, only valid for save_txt option")
parser.add_argument('--lr', type=float, default=0.0003, help="Learning rate for training")
parser.add_argument('--epoch', type=int, default=5, help="Number of training epochs")
parser.add_argument('--student_layers', type=int, default=4, help="Number of student model layers to use")
parser.add_argument('--ckpt_path', type=str, default="./ckpt/model", help="Path to save checkpoints")
parser.add_argument('--load_student_ckpt_path', type=str, help="Path to load student checkpoints")
parser.add_argument('--node_per_gpu', type=int, default=6, help="The number of gpus per node")
parser.add_argument('--world_size', type=int, default=3, help="Total number of GPUs")
parser.add_argument('--count_path', type=str, default="./wikitext/wikitext_count.pkl", help="Path to count data pickle file")
args = parser.parse_args()

#  Dataset for token classification
class TokenClassificationDataset(Dataset):
    def __init__(self, args, sample_size=10000):
        # Load data from a JSON file
        with open(args.json_path, 'r', encoding='utf-8') as file:
            self.original_data = json.load(file)
   
        self.sample_size = sample_size
        self.data = self.original_data
        self.tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
        self.max_length = 512
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt = self.data[idx]
        return prompt

    def collate_fn(self, sentences):
        data = self.tokenize(sentences)
        return data

    def tokenize(self, texts):
        # modified from ./llm2vec/llm2vec.py
        texts_2 = []
        original_texts = []
        
        for text in texts:
            t = text.split("!@#$%^&*()")
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(len(ids["input_ids"][0]))
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(len(ids["input_ids"][0]))
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)
        original["embed_mask"] = embed_mask

        return original


import pickle
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

save_dir = args.save_dir

# Load sense dictionary if a path is provided
sense_dict = None
if args.sense_dict_path is not None:
    with open(args.sense_dict_path, 'rb') as handle:
        sense_dict = pickle.load(handle)

sense_func = args.sense_func
run_option = args.run_option
student_layers = args.student_layers
task_name = args.task_name
node_per_gpu=args.node_per_gpu
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42) 

# Tain the student model using distributed training
def train_model(rank, world_size, dataset, args):
    save_interval = 3600  # Save a checkpoint every hour 
    last_save_time = time.time()  
    model_meta = mteb.get_model_meta(args.model_name)
    teacher_model_info = {
        'base_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
        'peft_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',
        'device_map': None,
        'torch_dtype': torch.bfloat16
    }
    student_model_info = {
        'base_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
        'peft_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',
        'device_map': None,
        'torch_dtype': torch.bfloat16
    }
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  
        world_size=world_size,
        rank=rank,
    )
    # For each node, we need half of them to run on teacher model, and half of them to run on student model.
    teacher_device_start_indice = node_per_gpu //2 
    local_rank_cal = rank % teacher_device_start_indice
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    print(f"Rank {rank}: teacher local rank {local_rank}, student device {local_rank+teacher_device_start_indice}, world size {world_size}")

    assert local_rank == local_rank_cal, f"Assertion failed: local_rank {local_rank} != local_rank_cal {local_rank_cal}"
    
    # Assign GPUs: teacher model uses GPUs starting from index local_rank+teacher_device_start_indice, student uses GPU at local_rank
    teacher_device = torch.device(f"cuda:{local_rank+ teacher_device_start_indice}")  # e.g., GPUs 3,4,5
    student_device = torch.device(f"cuda:{local_rank }")    # e.g., GPUs 0,1,2

    # Initializate teacher models
    teacher_model = TeacherWrapper(**teacher_model_info)
    teacher_model.mteb_model_meta = model_meta 

    # Freeze teacher model parameters
    for param in teacher_model.model.parameters():
        param.requires_grad = False
    teacher_model.model.set_device(teacher_device)

    # Initialize student models
    student_model = StudentWrapper(**student_model_info)
    student_model.mteb_model_meta = model_meta 
    # Load checkpoint if available
    if os.path.exists(args.load_student_ckpt_path):
        print("Load ckpt from", args.load_student_ckpt_path)
        load_checkpoint = True
    else:
        load_checkpoint = False
        print("No ckpt found, start from scratch")
    if load_checkpoint:
        checkpoint = torch.load(args.load_student_ckpt_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        # Remove DDP wrapper prefix if present in checkpoint keys
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        student_model.model.load_state_dict(new_state_dict)


    student_model.model.set_device(student_device)
    for name, param in student_model.model.named_parameters():
        param.requires_grad = True
    student_model.model = DDP(student_model.model, device_ids=[student_device.index])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    # Adjust learning rate based on GPUs
    lr = args.lr * args.world_size 
    optimizer = optim.AdamW(student_model.model.parameters(), lr=lr, weight_decay=1e-4) 
    batch_size = 3
    num_training_steps = len(dataset) / world_size / batch_size * args.epoch
    num_warmup_steps = 300  # Warmup steps
    
    scheduler = get_scheduler(
        name="cosine",  # Cosine schedule with warmup and decay
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # If resuming from a checkpoint, load optimizer and scheduler states
    if load_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        start_batch_idx = checkpoint['batch_idx']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, batch {start_batch_idx}")
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    else:
        start_epoch = 0
        start_batch_idx = 0

    # Ensure student model parameters remain trainable
    for param in student_model.model.parameters():
        param.requires_grad = True

    pad_id = dataset.tokenizer.pad_token_id
    # Training loop over epochs
    for epoch in tqdm(range(start_epoch, args.epoch)):
        print(f"epoch: {epoch}")
        total_correct = 0
        total_count = 0
        sampler.set_epoch(epoch)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=dataset.collate_fn)
        for batch_idx, features in enumerate(tqdm(data_loader)):
            # Skip already processed batches when resuming
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue  
            batch_correct = 0
            batch_count = 0
            batch_loss = 0
            optimizer.zero_grad()


            student_ids, embed_mask, student_outputs = student_model.model.module.train_model(features)
            teacher_ids, embed_mask, teacher_outputs = teacher_model.model.evaluate_model(features)
            teacher_ids = teacher_ids.to(student_ids.device)
            teacher_outputs = teacher_outputs.to(student_outputs.device)
            # Ensure the token ids match between teacher and student
            assert torch.equal(student_ids, teacher_ids), "Two ids are not equal!"

            # Iterate over each token in the batch to compute loss token-wise
            for i in range(student_outputs.shape[0]):
                for j in range(student_outputs.shape[1]):
                    id = student_ids[i][j].item()
                    if id == pad_id:
                        continue  # Skip padding tokens
                    if not (id in sense_dict):
                        print(id, "not in")
                        continue
                    token_emb = sense_dict[id].to(student_outputs.device)
                    student_logit = torch.matmul(token_emb, student_outputs[i, j, :])
                    student_target = torch.argmax(student_logit)
                    teacher_logit = torch.matmul(token_emb, teacher_outputs[i, j, :])
                    teacher_target = torch.argmax(teacher_logit)
                    loss = loss_fn(student_logit, teacher_target)
                    batch_loss += loss
                    batch_count += 1
                    if torch.equal(student_target, teacher_target):
                        batch_correct += 1


            batch_loss_back = batch_loss / batch_count
            batch_loss_back.backward()
            optimizer.step()
            scheduler.step()

            batch_loss_tensor = batch_loss.detach().clone()
            batch_count_tensor = torch.tensor(batch_count, dtype=torch.float32, device=student_device)
            batch_correct_tensor = torch.tensor(batch_correct, dtype=torch.float32, device=student_device)
            dist.all_reduce(batch_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_correct_tensor, op=dist.ReduceOp.SUM)
            batch_loss_mean = batch_loss_tensor / batch_count_tensor
            batch_correct_mean = batch_correct_tensor / batch_count_tensor
            if local_rank == 0:
                if batch_idx % 10 == 0:
                    print(f"length {batch_count}/{batch_count_tensor} Loss {batch_loss_back:.2f}/{batch_loss_mean:.2f} Acc {batch_correct/batch_count:.2f}/{batch_correct_mean:.2f}")
            total_correct += batch_correct_tensor
            total_count += batch_count_tensor
            
            current_time = time.time()
            # Save checkpoint 
            if current_time - last_save_time >= save_interval:
                last_save_time = current_time
                if rank == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': student_model.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(), 
                        'loss': batch_loss
                    }
                    torch.save(checkpoint, args.ckpt_path + time.strftime("%Y%m%d_%H%M%S") + ".pth")
                    print(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}")
                    print(f"epoch acc {total_correct/total_count:.2f}")
                torch.distributed.barrier()
        
        # Save checkpoint at the end of each epoch
        if rank == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': student_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(), 
                'loss': loss
            }
            torch.save(checkpoint, args.ckpt_path + str(epoch) + ".pth")
            print(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}")
            print(f"epoch acc {total_correct/total_count:.2f}")
        torch.distributed.barrier()

# Function to gather token embeddings
def gather_embedding(rank, world_size, dataset, args):
    # Load count data that tracks frequency of token ids
    with open(args.count_path, 'rb') as handle:
        count_data = pickle.load(handle)

    model_meta = mteb.get_model_meta(args.model_name)

    teacher_model_info = {
        'base_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
        'peft_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',
        'device_map': None,
        'torch_dtype': torch.bfloat16
    }

    dist.init_process_group(
        backend="nccl",
        init_method="env://", 
        world_size=world_size,
        rank=rank,
    )
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    print(f"Rank {rank}: teacher local rank {local_rank}, world size {world_size}")

    teacher_device = torch.device(f"cuda:{local_rank}") 
    teacher_model = TeacherWrapper(**teacher_model_info)
    teacher_model.mteb_model_meta = model_meta 

    # Freeze teacher model parameters for evaluation
    for param in teacher_model.model.parameters():
        param.requires_grad = False
    teacher_model.model.set_device(teacher_device)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    batch_size = 16
    sampler.set_epoch(0)
    embedding_nested_dict = dict()
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=dataset.collate_fn)
  
    for batch_idx, features in enumerate(tqdm(data_loader)):
        teacher_ids, attention_mask, teacher_outputs = teacher_model.model.evaluate_model(features)
        teacher_outputs = teacher_outputs.cpu().float().numpy()
        teacher_ids = teacher_ids.to("cpu")
        # For each token in the batch, save its embedding 
        for i in range(teacher_ids.shape[0]):
            for j in range(teacher_ids.shape[1]):
                if attention_mask[i, j] == 0:
                    continue 
                input_id = teacher_ids[i, j].item()
                if input_id not in count_data:
                    print(input_id, "not in count data")
                    continue
                random_number = random.randint(0, count_data[input_id])
                # Only keep embeddings for tokens that pass the threshold
                if random_number < 1000:
                    embedding_nested_dict.setdefault(input_id, [])
                    embedding_nested_dict[input_id].append(teacher_outputs[i, j, :])

    # Convert each token embeddings into numpy arrays
    for key in embedding_nested_dict.keys():
        embedding_nested_dict[key] = np.stack(embedding_nested_dict[key])      

    with open(args.save_dir + args.task_name + "_" + str(local_rank) + ".pkl", 'wb') as handle:
        pickle.dump(embedding_nested_dict, handle)

def main(args):
    if run_option == "train":
        # Distributed training mode
        rank = int(os.getenv("RANK", "0"))
        world_size = args.world_size
        dataset = TokenClassificationDataset(args)
        train_model(rank, world_size, dataset, args)
    elif run_option == "gatheremb":
        # Gather embeddings mode using the teacher model
        rank = int(os.getenv("RANK", "0"))
        world_size = args.world_size
        dataset = TokenClassificationDataset(args)
        gather_embedding(rank, world_size, dataset, args)
    elif args.run_option == "count":  
        # Count mode: count the frequency of each token id in the dataset
        world_size = 1
        dataset = TokenClassificationDataset(args)
        data_loader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)
        count_dict = dict()
        for batch_idx, features in enumerate(tqdm(data_loader)):
            for i in range(features["input_ids"].shape[0]):
                for j in range(features["input_ids"].shape[1]):
                    id = features["input_ids"][i][j].item()
                    if not (id in count_dict):
                        count_dict[id] = 0
                    count_dict[id] += 1
        with open(args.save_dir + args.task_name + "_count.countpkl", 'wb') as handle:
            pickle.dump(count_dict, handle)

    elif run_option == "replace":  # Evaluation mode using student model replacement
        assert args.run_option == "replace"
        model_meta = mteb.get_model_meta(args.model_name)
        from mteb.models.student_models import StudentWrapper
    
        student_model_info = {
            'base_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
            'peft_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',
            'device_map': None,
            'torch_dtype': torch.bfloat16
        }
    
        student_model = StudentWrapper(**student_model_info)
        student_model.mteb_model_meta = model_meta 
    
        ckpt_path = args.ckpt_path
        print("Load ckpt from", ckpt_path)
        
        model_state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        # Remove DDP prefix if present in checkpoint keys
        for key, value in model_state_dict['model_state_dict'].items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
    
        student_model.model.load_state_dict(new_state_dict)
        for param in student_model.model.parameters():
            param.requires_grad = False
    
        tasks = mteb.get_tasks(tasks=[args.task_name])
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(student_model, output_folder=args.output_dir)

    elif run_option == "teacher_replace" or run_option == "cluster" or run_option == "llama":  # Evaluation modes using teacher model
        model_meta = mteb.get_model_meta(args.model_name)
        from mteb.models.teacher_models import TeacherWrapper
    
        student_model_info = {
            'base_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
            'peft_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',
            'device_map': None,
            'torch_dtype': torch.bfloat16
        }
    
        teacher_model = TeacherWrapper(**student_model_info)
        teacher_model.mteb_model_meta = model_meta 
        for param in teacher_model.model.model.parameters():
            param.requires_grad = False
    
        tasks = mteb.get_tasks(tasks=[args.task_name])
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(teacher_model, output_folder=args.output_dir)
    
    elif run_option == "save_txt":  # Evaluation mode to save textual outputs
        model_meta = mteb.get_model_meta(args.model_name)
        from mteb.models.student_models import StudentWrapper
        student_model_info = {
            'base_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
            'peft_model_name_or_path': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',
            'device_map': None,
            'torch_dtype': torch.bfloat16
        }
    
        student_model = StudentWrapper(**student_model_info)
        student_model.mteb_model_meta = model_meta 
        for param in student_model.model.model.parameters():
            param.requires_grad = False
    
        tasks = mteb.get_tasks(tasks=[args.task_name])
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(student_model, output_folder=args.output_dir)


if __name__ == "__main__":
    main(args)
