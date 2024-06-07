export HF_HOME='/path/to/data/huggingface'

export exp_name="llava16_7b_cal"  
export base_dir="/path/llava_v15/"

export finetune_json="/path/llava_v1_5_mix665k.json"
export finetune_imagedir="/path/data/"

bash scripts/general/run.sh
