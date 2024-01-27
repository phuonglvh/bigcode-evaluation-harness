python utils/generations_to_codexglue_codebleu.py \
    --generations_path runpod/Mistral-7B-v0.1/Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-generations_multiple-py.json \
    --predictions_path ./results/codebleu/Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-predictions_multiple-py.txt

python utils/generations_to_codexglue_bleu.py \
    --generations_path runpod/Mistral-7B-v0.1/Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-generations_multiple-py.json \
    --predictions_path ./results/bleu/Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-predictions_multiple-py.txt

python utils/human_eval_x_to_codexglue_codebleu.py \
    -lang python \
    --predictions_path runpod/Mistral-7B-v0.1/Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-generations_multiple-py.json \
    --references_path ./datasets/references/codebleu/Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-references_multiple-py.text

python utils/human_eval_x_to_codexglue_bleu.py \
    -lang python \
    --predictions_path runpod/Mistral-7B-v0.1/Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-generations_multiple-py.json \
    --references_path ./datasets/references/bleu/Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-references_multiple-py.jsonl

python main.py --model "mistralai/Mistral-7B-v0.1" \
    --tasks multiple-py \
    --max_length_generation 512 \
    --temperature 0.8 \
    --top_p 0.95 \
    --n_samples 200 \
    --batch_size 50 \
    --precision bf16 \
    --allow_code_execution \
    --trust_remote_code \
    --load_generations_path "./Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-generations_multiple-py.json" \
    --metric_output_path "./Mistral-7B-v0.1-temp0.8-p0.95-bf16-n200-batch100-maxlen512-py-generations_multiple-py-evaluation_results.json" \
    --use_auth_token
