# export HF_HOME='/path/to/data/huggingface'

export exp_name="llava16_7b_cal" 
export model_max_length="4096"
export base_dir="/path/llava_v15/"

export pretrain_llm="lmsys/vicuna-13b-v1.5"
export pretrain_json="/path/to/blip_laion_cc_sbu_558k.json"
export pretrain_imagedir="/path/data/"
export pretrain_image_aspect_ratio="anyres"
export pretrain_mm_patch_merge_type="spatial_unpad"

export finetune_image_aspect_ratio="anyres"
export finetune_mm_patch_merge_type="spatial_unpad"
export finetune_openvit="True"
export finetune_json="/path/llava_v1_6.json"
export finetune_imagedir="/path/data/"

bash scripts/general/run.sh
