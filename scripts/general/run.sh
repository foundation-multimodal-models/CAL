bash install.sh

# root dir
export base_dir=${base_dir:-"./"}

# exp param
export exp_name=${exp_name:-"llava_debug"} 
export expdetail=${expdetail:-"exp_detail"}
export mm_projector_type=${mm_projector_type:-"mlp2x_gelu"} # mlp2x_gelu, cabs
export vision_tower=${vision_tower:-"openai/clip-vit-large-patch14-336"}
export model_max_length=${model_max_length:-"2048"} # 4096(1.6)

# pretrain param
export pretrain_llm=${pretrain_llm:-"lmsys/vicuna-7b-v1.5"} # lmsys/vicuna-7b-v1.5, lmsys/vicuna-13b-v1.5
export pretrain_openvit=${pretrain_openvit:-"False"}
export pretrain_image_aspect_ratio=${pretrain_image_aspect_ratio:-"square"} # pad, square, anyres(1.6)
export pretrain_mm_patch_merge_type=${pretrain_mm_patch_merge_type:-"flat"} # flat, spatial_unpad(1.6)
export pretrain_save_dir=${pretrain_save_dir:-"${base_dir}${exp_name}/${exp_name}_pretrain"}
export pretrain_conv_version=${pretrain_conv_version:-"plain"} # plain, v1
export pretrain_json=${pretrain_json:-"/blip_laion_cc_sbu_558k.json"}
export pretrain_imagedir=${pretrain_imagedir:-"/path/images"}
export pretrain_deepspeed=${pretrain_deepspeed:-"./scripts/zero2.json"}
export pretrain_wandb_project=${pretrain_wandb_project:-"llava_$(basename $pretrain_llm)_pretrain"}
export pretrain_wandb_process=${pretrain_wandb_process:-"${exp_name}_pretrain"}
export pretrain_total_batchsize=${pretrain_total_batchsize:-"256"}
export pretrain_grad_acumsteps=${pretrain_grad_acumsteps:-"1"}
export pretrain_num_epoch=${pretrain_num_epoch:-"1"}
export pretrain_lr=${pretrain_lr:-"1e-3"}

# finetune param
export finetune_llm=${finetune_llm:-"$pretrain_save_dir"}
export finetune_openvit=${finetune_openvit:-"False"}
export finetune_image_aspect_ratio=${finetune_image_aspect_ratio:-"pad"} # pad, square, anyres(1.6)
export finetune_mm_patch_merge_type=${finetune_mm_patch_merge_type:-"flat"} # flat, spatial_unpad(1.6)
export finetune_save_dir=${finetune_save_dir:-"${base_dir}${exp_name}/${exp_name}_finetune"}
export fintune_conv_version=${fintune_conv_version:-"v1"} # plain, v1
export finetune_json=${finetune_json:-"/path/llava_v1_5_mix665k.json"}
export finetune_imagedir=${finetune_imagedir:-"/path/data/"}
export finetune_deepspeed=${finetune_deepspeed:-"./scripts/zero3.json"}
export finetune_wandb_project=${finetune_wandb_project:-"llava_$(basename $pretrain_llm)_finetune"}
export finetune_wandb_process=${finetune_wandb_process:-"${exp_name}_finetune"}
export finetune_total_batchsize=${finetune_total_batchsize:-"128"}
export finetune_grad_acumsteps=${finetune_grad_acumsteps:-"1"}
export finetune_num_epoch=${finetune_num_epoch:-"1"}
export finetune_lr=${finetune_lr:-"2e-5"}

# exp param
echo "exp_name: $exp_name"
echo "mm_projector_type: $mm_projector_type"
echo "vision_tower: $vision_tower"
echo "model_max_length: $model_max_length"
echo

# pretrain param
echo "pretrain_llm: $pretrain_llm"
echo "pretrain_openvit: $pretrain_openvit"
echo "pretrain_image_aspect_ratio: $pretrain_image_aspect_ratio"
echo "pretrain_mm_patch_merge_type: $pretrain_mm_patch_merge_type"
echo "pretrain_save_dir: $pretrain_save_dir"
echo "pretrain_conv_version: $pretrain_conv_version"
echo "pretrain_json: $pretrain_json"
echo "pretrain_imagedir: $pretrain_imagedir"
echo "pretrain_deepspeed: $pretrain_deepspeed"
echo "pretrain_wandb_project: $pretrain_wandb_project"
echo "pretrain_wandb_process: $pretrain_wandb_process"
echo "pretrain_total_batchsize: $pretrain_total_batchsize"
echo "pretrain_grad_acumsteps: $pretrain_grad_acumsteps"
echo "pretrain_num_epoch: $pretrain_num_epoch"
echo "pretrain_lr: $pretrain_lr"
echo

# finetune param
echo "finetune_llm: $finetune_llm"
echo "finetune_openvit: $finetune_openvit"
echo "finetune_image_aspect_ratio: $finetune_image_aspect_ratio"
echo "finetune_mm_patch_merge_type: $finetune_mm_patch_merge_type"
echo "finetune_save_dir: $finetune_save_dir"
echo "fintune_conv_version: $fintune_conv_version"
echo "finetune_json: $finetune_json"
echo "finetune_imagedir: $finetune_imagedir"
echo "finetune_deepspeed: $finetune_deepspeed"
echo "finetune_wandb_project: $finetune_wandb_project"
echo "finetune_wandb_process: $finetune_wandb_process"
echo "finetune_total_batchsize: $finetune_total_batchsize"
echo "finetune_grad_acumsteps: $finetune_grad_acumsteps"
echo "finetune_num_epoch: $finetune_num_epoch"
echo "finetune_lr: $finetune_lr"

sudo mkdir -p "$base_dir"
sudo chmod 777 "$base_dir"

if [ -n "$skip_pretrain" ] ;then
    echo "Skip pretrain, make sure you know what you are doing"
else
    echo "Start pretrain"
    bash scripts/general/pretrain.sh 
fi

if [ -n "$skip_finetune" ] ;then
    echo "Skip finetune, make sure you know what you are doing"
else
    echo "Start finetune"
    bash scripts/general/finetune.sh
fi

source ./unset.sh 
