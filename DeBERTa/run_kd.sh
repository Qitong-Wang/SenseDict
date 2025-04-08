
task=$1
export PYTHONPATH=$(pwd)

echo "-------------------"
echo $task
echo "-------------------"


case $task in 
    CoLA)
    parameters="--num_train_epochs 10 \
    --fp16 True \
    --warmup 50 \
    --learning_rate 5.5e-6 \
    --train_batch_size 32 \
    --cls_drop_out 0.1 "
    ;;
    MNLI)
    parameters="--num_train_epochs 2 \
    --fp16 True \
    --warmup 500 \
    --learning_rate 7e-6 \
    --train_batch_size 64 \
    --cls_drop_out 0.3 \
    --max_seq_len 256 \
    --eval_batch_size 256 \
    --dump_interval 1000 "
    ;;
    MRPC)
    parameters="--num_train_epochs 6 \
    --fp16 True \
    --warmup 50 \
    --learning_rate 6e-6  \
    --train_batch_size 32 \
    --max_seq_len 128     \
    --cls_drop_out 0.2  "
    ;;
    QNLI)
    parameters="--num_train_epochs 6 \
    --fp16 True \
    --warmup 200 \
    --learning_rate 7e-6  \
    --train_batch_size 64 \
    --max_seq_len 512     \
    --cls_drop_out 0.2 "
    ;;
    QQP)
    parameters="--num_train_epochs 6 \
    --fp16 True \
    --warmup 1000 \
    --max_seq_len 320 \
    --learning_rate 1e-5  \
    --train_batch_size 64 \
    --cls_drop_out 0.2 "
    ;;
    RTE)
    parameters="--num_train_epochs 6 \
    --fp16 True \
    --warmup 50 \
    --learning_rate 1e-5  \
    --train_batch_size 32 \
    --max_seq_len 320     \
    --cls_drop_out 0.3 "
    ;;
    SST-2)
    parameters="--num_train_epochs 8 \
    --fp16 True \
    --warmup 500 \
    --learning_rate 6e-6  \
    --train_batch_size 32 \
    --cls_drop_out 0.1  \
    --max_seq_len 128 "
    ;;
    STS-B)
    parameters="--num_train_epochs 4 \
    --fp16 True \
    --warmup 50 \
    --learning_rate 7e-6  \
    --train_batch_size 32 \
    --cls_drop_out 0.1 \
    --max_seq_len 128 "
    ;;

esac
# Train the teacher model first
python a0a_run_teachermodel.py \
--model_config ./experiments/glue/config.json  \
--tag deberta-v3-large \
--do_train \
--task_name $task \
--data_dir ./glue/$task \
--init_model deberta-v3-large \
--output_dir ./ckpts/teacher-v3-large/$task   \
$parameters 



# Modify the checkpoints of fine-tuned DeBERTa model.
case $task in
  CoLA)
  teacher_ckpt_path="./ckpts/teacher-v3-large/CoLA/pytorch.model-001072.bin"
  ;;
  MNLI)
  teacher_ckpt_path="./ckpts/teacher-v3-large/MNLI/pytorch.model-011000.bin"
  ;;
  MRPC)
  teacher_ckpt_path="./ckpts/teacher-v3-large/MRPC/pytorch.model-000460.bin"
  ;;
  QNLI)
  teacher_ckpt_path="./ckpts/teacher-v3-large/QNLI/pytorch.model-003000.bin"
  ;;
  QQP)
  teacher_ckpt_path="./ckpts/teacher-v3-large/QQP/pytorch.model-017000.bin"
  ;;
  RTE)
  teacher_ckpt_path="./ckpts/teacher-v3-large/RTE/pytorch.model-000312.bin"
  ;;
  SST-2)
  teacher_ckpt_path="./ckpts/teacher-v3-large/SST-2/pytorch.model-012000.bin"
  ;;
  STS-B)
  teacher_ckpt_path="./ckpts/teacher-v3-large/STS-B/pytorch.model-000718.bin"
  ;;
esac


case $task in
  MNLI)
  num_labels=3
  ;;
  STS-B)
  num_labels=1
  ;;
  *)
  num_labels=2
esac

# Train student

python a1_gather_sense.py --json_file ./gather_json/all_train.json --count_file ./resources/all_train_count.countpkl --output_file ./resources/all_train_""$task""_1000.combinepkl  --teacher_ckpt_path $teacher_ckpt_path --k 1000 --num_labels $num_labels
python a1_mclsingle.py --input_file  ./resources/all_train_""$task""_1000.combinepkl --output_file ./resources/all_train_""$task"".kmeanspkl --distance 20 --metric kmeans




case $2 in 
  train)
  json_path="./resources/glue_json/""$task""_train.json"
  ;;
  alltrain)
  json_path="./resources/glue_json/all_train.json"
  ;;

esac


# Train student


torchrun  --nnodes=1 --nproc_per_node=5  a3_traincluster_evalmulti.py  --json_path $json_path --cluster_path "./resources/wiki_""$task"".kmeanspkl" --teacher_ckpt_path  $teacher_ckpt_path --ckpt_path "./ckpts/train_student/$2/""$task""_.ckpt" --num_labels $num_labels --epoch 15 --lr 0.0005  

# Evaluation
python a0b_run_studentmodel.py \
--model_config ./experiments/glue/config.json  \
--tag deberta-v3-xsmall \
--do_eval \
--task_name $task \
--data_dir ./glue/$task \
--init_model deberta-v3-xsmall \
--output_dir ./results/student-v3-xsmall/$task   \
$parameters \
--student_ckpt_path "./ckpts/train_student/$2/""$task""_14.ckpt" \
--cluster_path "./resources/wiki_""$task"".kmeanspkl"
