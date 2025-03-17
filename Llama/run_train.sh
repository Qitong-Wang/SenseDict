export PYTHONPATH=$(pwd)

task_name=$1



task_names=(
  "ArxivClassification"  #1
  "Banking77Classification"
  "DBpediaClassification" #3
  "EmotionClassification"
  "FinancialPhrasebankClassification" #5
  "FrenkEnClassification"
  "ImdbClassification" #7
  "CanadaTaxCourtOutcomesLegalBenchClassification"
  "NewsClassification" #9
  "PatentClassification" 
  "PoemSentimentClassification" #11
  "ToxicChatClassification"
  "ToxicConversationsClassification" #13
  "TweetSentimentExtractionClassification"
  "TweetTopicSingleClassification" #15
  "YelpReviewFullClassification"
)
# Validate input
if [[ $1 -ge 1 && $1 -le ${#task_names[@]} ]]; then
  task_name=${task_names[$(( $1 - 1 ))]}
  echo "The task name for \$1=$1 is: $task_name"
else
  echo "Invalid input. Please enter a number between 1 and ${#task_names[@]}."
fi





echo "----------------------------------------------------"
echo  $task_name 
echo "----------------------------------------------------"



CKPT_PATH=./ckpt/$task_name/


mkdir -p $CKPT_PATH

torchrun --nproc_per_node=3 --nnodes=3  --master_addr=$2 --master_port=$3 --node_rank=$4 a2_train_multi.py --model_name McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
--task_name $task_name \
--task_to_instructions_fp test_configs/mteb/task_to_instructions.json \
--sense_func dot \
--run_option train \
--lr 0.000005  \
--student_layers 4  \
--json_path ./text_train/$task_name/train.json \
--epoch 5 \
--ckpt_path $CKPT_PATH \
--world_size 9



# if $4 (node rank) is 0, evaluate the model
if [ "$4" -eq 0 ]; then

echo "----------------------------------------------------"
echo "Evaluate"
echo "----------------------------------------------------"



echo "----------------------------------------------------"
echo "Epoch " $epoch
echo "----------------------------------------------------"



OUTPUT_PATH=./result/$epoch/
mkdir -p $OUTPUT_PATH

PRINT_PATH=$OUTPUT_PATH"McGill-NLP__LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/no_revision_available/""$task_name"".json"
rm -rf $PRINT_PATH

python a2_train_multi.py --model_name McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
--task_name $task_name \
--task_to_instructions_fp test_configs/mteb/task_to_instructions.json \
--output_dir $OUTPUT_PATH \
--sense_func dot \
--student_layers 4  \
--epoch $epoch \
--ckpt_path $CKPT_PATH/$epoch.pth \
--run_option replace  


cat $PRINT_PATH

done

python s1_print_result.py --folder_path  ./result/
