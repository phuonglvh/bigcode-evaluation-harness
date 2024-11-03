AUTHOR="codellama"
MODEL_NAME="CodeLlama-13b-Python-hf"

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

# 2
num_beams=2
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# 3
num_beams=3
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# 4
num_beams=4
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# 5
num_beams=5
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# 6
num_beams=6
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# 8
num_beams=8
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# 10
num_beams=10
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# 15
num_beams=15
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# 20
num_beams=20
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

# execute
BASE_DIR=./runpod/$MODEL_NAME/$lang/do_sample$do_sample
mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
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

# compute EM, BLEU, CodeBLEU
full_language=java
generations_path="$(realpath $BASE_DIR)/$generations_name.json"

# BLEU score
bleu_predictions_path="$(realpath $BASE_DIR)/$common_name-bleu-predictions_multiple-$lang.txt"
python utils/generations_to_codexglue_bleu.py \
    --load_generations_path "$generations_path" \
    --save_predictions_format_path "$bleu_predictions_path"

bleu_references_path="$(realpath $BASE_DIR)/$common_name-bleu-references_multiple-$lang.jsonl"
python utils/human_eval_x_to_codexglue_bleu.py \
    --language $full_language \
    --load_generations_path "$generations_path" \
    --save_references_path "$bleu_references_path"
(
    cd ./CodeXGLUE/Text-Code/text-to-code || ! echo "cd failure"
    python evaluator/evaluator.py --answers "$bleu_references_path" --predictions "$bleu_predictions_path"
)

# CodeBLEU score
codebleu_predictions_path="$(realpath $BASE_DIR)/$common_name-codebleu-predictions_multiple-$lang.txt)"
python utils/generations_to_codexglue_codebleu.py \
    --load_generations_path "$generations_path" \
    --save_predictions_format_path "$codebleu_predictions_path"

codebleu_references_path="$(realpath $BASE_DIR)/$common_name-codebleu-references_multiple-$lang.txt)"
python utils/human_eval_x_to_codexglue_codebleu.py \
    --language $full_language \
    --load_generations_path "$generations_path" \
    --save_references_path "$codebleu_references_path"

(
    cd ./CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU || ! echo "cd failure"
    python calc_code_bleu.py \
        --lang $full_language \
        --params "0.25,0.25,0.25,0.25" \
        --refs "$codebleu_references_path" \
        --hyp "$codebleu_predictions_path"
)