#!/bin/bash

# 激活虚拟环境
source activate topo_qwen

# 指定使用的GPU编号
export CUDA_VISIBLE_DEVICES=1,2,3

# 创建日志文件，文件名为当前日期
log_file="logs/llamafactory-cli_train_$(date +'%Y-%m-%d-%H-%M-%S').log"

# 确保日志目录存在
mkdir -p "$(dirname "$log_file")"

# 运行命令并记录日志
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /share/huggingface/Qwen2-VL-7B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --dataset topo2text \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2-VL-7B-Instruct/lora/train_2024-12-28-03-13-08 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all >"$log_file" 2>&1

echo "Training process has been logged to $log_file"
