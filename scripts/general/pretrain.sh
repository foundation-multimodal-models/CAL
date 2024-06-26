#!/bin/bash
torchrun --nnodes=$ARNOLD_WORKER_NUM \
         --node_rank=$ARNOLD_ID \
         --nproc_per_node=$ARNOLD_WORKER_GPU \
         --master_addr=$ARNOLD_WORKER_0_HOST \
         --master_port=$port \
    llava/train/train_mem.py \
    --deepspeed "$pretrain_deepspeed" \
    --model_name_or_path "$pretrain_llm" \
    --version "$pretrain_conv_version" \
    --data_path "$pretrain_json" \
    --image_folder "$pretrain_imagedir" \
    --vision_tower "$vision_tower" \
    --open_vit "$pretrain_openvit" \
    --mm_projector_type "$mm_projector_type" \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio $pretrain_image_aspect_ratio \
    --mm_patch_merge_type $pretrain_mm_patch_merge_type \
    --bf16 True \
    --output_dir "$pretrain_save_dir" \
    --num_train_epochs $pretrain_num_epoch \
    --per_device_train_batch_size $[$pretrain_total_batchsize/$ARNOLD_WORKER_NUM/$ARNOLD_WORKER_GPU/$pretrain_grad_acumsteps] \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $pretrain_grad_acumsteps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate "$pretrain_lr" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $model_max_length \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --wandb_project "$pretrain_wandb_project" \
    --wandb_process "$pretrain_wandb_process"