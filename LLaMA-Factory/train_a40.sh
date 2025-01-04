#!/bin/bash

source activate topo_qwen

<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=0,1,3,4

# model_name_or_path="/share/huggingface/Qwen2-VL-2B-Instruct"
# model_name="Qwen2-VL-2B-Instruct"
model_name_or_path="/share/huggingface/Qwen2-VL-2B-Instruct"
=======
export CUDA_VISIBLE_DEVICES=1

model_name_or_path="/share/model/Qwen2-VL-2B-Instruct"
>>>>>>> d37fbace87fe0eefe1481f986515160256fa81c2
model_name="Qwen2-VL-2B-Instruct"
datasets="topo2text_v1,topo2text_v2"
dataset_dir="data"
output_base_dir="/share/topo_lab_res/$model_name/lora"

IFS=',' read -r -a dataset_array <<<"$datasets"

for dataset in "${dataset_array[@]}"; do
    res_name="train_${dataset}_$(date +'%Y-%m-%d-%H-%M-%S')"
<<<<<<< HEAD
    log_file="logs/llamafactory-cli_train_${dataset}_$(date +'%Y-%m-%d-%H-%M-%S').log"
=======
    log_file="logs/llamafactory-cli_train_${model_name}_${dataset}_$(date +'%Y-%m-%d-%H-%M-%S').log"
>>>>>>> d37fbace87fe0eefe1481f986515160256fa81c2
    output_dir="$output_base_dir/$res_name"

    mkdir -p "$(dirname "$log_file")"
    mkdir -p "$output_dir"

    echo "Starting training for dataset: $dataset"

    llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path "$model_name_or_path" \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template qwen2_vl \
        --flash_attn auto \
        --dataset_dir "$dataset_dir" \
        --dataset "$dataset" \
        --cutoff_len 2048 \
        --learning_rate 5e-05 \
        --num_train_epochs 3.0 \
        --max_samples 100000 \
<<<<<<< HEAD
        --per_device_train_batch_size 5 \
=======
        --per_device_train_batch_size 2 \
>>>>>>> d37fbace87fe0eefe1481f986515160256fa81c2
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 5 \
        --save_steps 100 \
        --warmup_steps 0 \
        --packing False \
        --report_to none \
        --output_dir "$output_dir" \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --optim adamw_torch \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all >"$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "Training for dataset $dataset completed successfully. Logged to $log_file"
    else
        echo "Training for dataset $dataset failed. Check log file $log_file for details."
    fi
    echo "---------------------------------------------------------"
done

echo "All training processes completed."
