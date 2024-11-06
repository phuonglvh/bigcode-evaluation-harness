AUTHOR="mistralai"
MODEL_NAME="Mistral-7B-v0.1"

max_length=1024
do_sample=False
num_return_sequences=1
batch_size=$num_return_sequences
n_samples=1
seed=0
precision=bf16
lang=java

limit_start=0
limit=158
eval_limit_start=0
eval_limit=158

save_every_k_tasks=1 # after completing k dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

do_sample=False

# 1
num_beams=1
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --token \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --num_beams $num_beams \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --save_generations \
    --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
    --save_references \
    --limit_start $limit_start \
    --limit $limit \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
    --max_memory_per_gpu auto

# # 2
# num_beams=2
# common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
# generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
# mkdir -p $BASE_DIR
# rm -rf /tmp/* /var/tmp/*

# python main.py --model "$AUTHOR/$MODEL_NAME" \
#     --tasks multiple-$lang \
#     --token \
#     --max_length_generation $max_length \
#     --do_sample $do_sample \
#     --num_beams $num_beams \
#     --seed $seed \
#     --n_samples $n_samples \
#     --batch_size $batch_size \
#     --precision $precision \
#     --allow_code_execution \
#     --trust_remote_code \
#     --save_every_k_tasks $save_every_k_iterations \
#     --save_generations \
#     --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
#     --save_references \
#     --limit_start $limit_start \
#     --limit $limit \
#     --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
#     --max_memory_per_gpu auto

# 3
num_beams=3
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --token \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --num_beams $num_beams \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --save_generations \
    --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
    --save_references \
    --limit_start $limit_start \
    --limit $limit \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
    --max_memory_per_gpu auto

# # 4
# num_beams=4
# common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
# generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
# mkdir -p $BASE_DIR
# rm -rf /tmp/* /var/tmp/*

# python main.py --model "$AUTHOR/$MODEL_NAME" \
#     --tasks multiple-$lang \
#     --token \
#     --max_length_generation $max_length \
#     --do_sample $do_sample \
#     --num_beams $num_beams \
#     --seed $seed \
#     --n_samples $n_samples \
#     --batch_size $batch_size \
#     --precision $precision \
#     --allow_code_execution \
#     --trust_remote_code \
#     --save_every_k_tasks $save_every_k_iterations \
#     --save_generations \
#     --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
#     --save_references \
#     --limit_start $limit_start \
#     --limit $limit \
#     --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
#     --max_memory_per_gpu auto

# # 5
# num_beams=5
# common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
# generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
# mkdir -p $BASE_DIR
# rm -rf /tmp/* /var/tmp/*

# python main.py --model "$AUTHOR/$MODEL_NAME" \
#     --tasks multiple-$lang \
#     --token \
#     --max_length_generation $max_length \
#     --do_sample $do_sample \
#     --num_beams $num_beams \
#     --seed $seed \
#     --n_samples $n_samples \
#     --batch_size $batch_size \
#     --precision $precision \
#     --allow_code_execution \
#     --trust_remote_code \
#     --save_every_k_tasks $save_every_k_iterations \
#     --save_generations \
#     --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
#     --save_references \
#     --limit_start $limit_start \
#     --limit $limit \
#     --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
#     --max_memory_per_gpu auto

# 6
num_beams=6
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --token \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --num_beams $num_beams \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --save_generations \
    --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
    --save_references \
    --limit_start $limit_start \
    --limit $limit \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
    --max_memory_per_gpu auto

# 8
num_beams=8
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --token \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --num_beams $num_beams \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --save_generations \
    --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
    --save_references \
    --limit_start $limit_start \
    --limit $limit \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
    --max_memory_per_gpu auto

# 10
num_beams=10
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --token \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --num_beams $num_beams \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --save_generations \
    --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
    --save_references \
    --limit_start $limit_start \
    --limit $limit \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
    --max_memory_per_gpu auto

# 15
num_beams=15
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --token \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --num_beams $num_beams \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --save_generations \
    --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
    --save_references \
    --limit_start $limit_start \
    --limit $limit \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
    --max_memory_per_gpu auto

# 20
num_beams=20
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --token \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --num_beams $num_beams \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --save_generations \
    --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
    --save_references \
    --limit_start $limit_start \
    --limit $limit \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
    --max_memory_per_gpu auto