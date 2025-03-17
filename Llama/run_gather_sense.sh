xport PYTHONPATH=$(pwd)

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
echo "Task Name:" $task_name
echo "----------------------------------------------------"

echo "----------------------------------------------------"
echo "count"
echo "----------------------------------------------------"
rm -rf ./resulttmp/
mkdir -p ./sense_dict/$task_name/
python a2_train_multi.py --model_name McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
--task_name $task_name \
--task_to_instructions_fp test_configs/mteb/task_to_instructions.json \
--json_path ./text_train/$task_name/train.json \
--output_dir ./resulttmp/ \
--save_dir ./sense_dict/$task_name/ \
--run_option count  




echo "----------------------------------------------------"
echo "Gather Embeddings"
echo "----------------------------------------------------"


torchrun --nproc_per_node=6 --nnodes=1  a2_train_multi.py --model_name McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
--task_name $task_name \
--task_to_instructions_fp test_configs/mteb/task_to_instructions.json \
--run_option gatheremb \
--json_path  ./text_train/$task_name/train.json  \
--save_dir ./sense_dict/$task_name/ \
--count_path ./sense_dict/$task_name/${task_name}_count.countpkl \
--world_size 6



python a0_generate_combinepkl.py --folder_names ./sense_dict/$task_name/  --output_keyword ./sense_dict/combine_1000