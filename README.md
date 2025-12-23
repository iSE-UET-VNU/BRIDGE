# PubMedQA Evaluation Examples

This document shows how to run the enhanced evaluation script for PubMedQA with various options.

## Basic Usage

Run zero-shot evaluation on a small number of examples:
```bash
# Make sure OPENAI_API_KEY is set in your environment
export OPENAI_API_KEY=your_api_key_here

# Run zero-shot evaluation on 10 examples
python eval.py --model gpt-4o --max_samples 10 --output_dir results
```

## Few-Shot Learning Examples

Run evaluation with 3-shot examples:
```bash
python eval.py --model gpt-4o --max_samples 50 --n_shots 3 --output_dir results
```

## Saving Predictions to a Separate File

Save predictions to a separate file for further analysis:
```bash
python eval.py --model gpt-4o --max_samples 100 --n_shots 5 --output_dir results --predictions_file pubmedqa_preds.json
```

## Full Dataset Evaluation

Run on the full dataset with few-shot learning:
```bash
python eval.py --model gpt-4o --n_shots 5 --output_dir results --predictions_file pubmedqa_full_preds.json
```

## Comparing Different Models

Compare different models (running sequentially):
```bash
# GPT-4o
python eval.py --model gpt-4o --n_shots 3 --output_dir results --predictions_file pubmedqa_gpt4o_preds.json

# GPT-3.5-turbo
python eval.py --model gpt-3.5-turbo --n_shots 3 --output_dir results --predictions_file pubmedqa_gpt35_preds.json
```

## Notes

- The script uses exponential backoff with the `backoff` library to handle API rate limits
- Set the `--seed` parameter to ensure reproducible few-shot example selection
- Results are saved in JSON format for easy analysis

## HF Instruct Models (local) 🐥

Evaluate PubMedQA with a Hugging Face instruct model (mirrors the blackbox flow):

```bash
# Optional: authenticate to Hugging Face if the model is gated (e.g., Llama 2)
huggingface-cli login  # requires an access token with model access

# Zero-shot on 20 samples with Llama 2 7B Chat in 4-bit quantization
python evaluation/pubmedqa/eval_instruct.py \
	--model_id meta-llama/Llama-2-7b-chat-hf \
	--max_samples 20 \
	--n_shots 0 \
	--load_in_4bit \
	--max_new_tokens 16 \
	--temperature 0.0

# Few-shot (5-shot) on 100 samples, save raw predictions to a file
python evaluation/pubmedqa/eval_instruct.py \
	--model_id meta-llama/Llama-2-7b-chat-hf \
	--max_samples 100 \
	--n_shots 5 \
	--load_in_4bit \
	--predictions_file pubmedqa_preds_llama2_7b_chat_5shot.json

# If you don't have access to Llama 2, try an open instruct model (example):
python evaluation/pubmedqa/eval_instruct.py \
	--model_id mistralai/Mistral-7B-Instruct-v0.2 \
	--max_samples 50 \
	--n_shots 3 \
	--load_in_4bit
```

Tips:

- Use `--load_in_4bit` or `--load_in_8bit` to fit models on limited GPUs/CPUs.
- Results JSON is saved under `/results`, matching the blackbox script convention.
- The CLI supports `--device_map auto|cuda|cpu` and `--dtype auto|float16|bfloat16|float32` if you need control.


## Running SFT on the FOMC dataset

Usage:

# First generate the dataset:

python evaluation/fomc/sft_gpt4o_gen.py --output_path ./data/fomc_sft_dataset.json

# Then run SFT:

accelerate launch sft_fomc.py \
    --model_name_or_path bigscience/bloomz-560m \
    --dataset_name ./data/fomc_sft_dataset.json \
    --output_dir ./models/bloomz-560m-fomc \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 100 \
    --logging_steps 10 \
    --max_seq_length 512# BRIDGE
