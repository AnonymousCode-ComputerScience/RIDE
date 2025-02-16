#!/bin/bash
#SBATCH --job-name=urial
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000
#SBATCH --partition A100
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=**.**@**.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load anaconda/anaconda3
module load cuda/cuda-11.1.0
module load cudnn/cudnn-8.0.4
export CONDA_ENVS=/nfsdata/data/devinh/envs
source activate $CONDA_ENVS/alignment
cd /nfsdata/data/devinh/URIAL

#version=inst_1k_v4.help
#version=value_impact_rewrite_upgrade_icl_16_8_10.random_swap
#version=value_impact_rewrite_upgrade_icl_8_15_1
#version=value_impact_rewrite_twoturns_icl_16_8_10
#version=value_impact_rewrite_twoturns_icl_8_15_1
#version=value_impact_rewrite_upgrade_icl_16_8_safe
#version=value_impact_rewrite_upgrade_icl_8_15_1.reduced
#version=value_impact_rewrite_upgrade_icl_16_8_sorry7-variant
version=no_instruct

#model_name="meta-llama/Llama-2-7b-hf"
#model=Llama-2-7b-hf
model_name="meta-llama/Llama-2-7b-chat-hf"
model=Llama-2-7b-chat-hf
temp=0
rp=1.15
output_dir="result_dirs/mt-bench/urial_${version}/"
mkdir -p $output_dir
tps=1
gpu=0


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --model_name ${model_name} \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 4 --max_tokens 2048 \
    --filepath $output_dir/$model.turn1.json \
    --overwrite

CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --model_name ${model_name} \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/${model}.turn1.json \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/$model.turn2.json \
    --overwrite

python run_scripts/mt-bench/formatting_results.py ${model} ${version} ${output_dir}
mkdir -p FastChat/fastchat/llm_judge/data/mt_bench/model_answer/
cp ${output_dir}/${model}-${version}_reformat.jsonl FastChat/fastchat/llm_judge/data/mt_bench/model_answer/
cd FastChat/fastchat/llm_judge/
python gen_judgment.py --model-list ${model}-${version}_reformat --parallel 8
cd /nfsdata/data/devinh/URIAL
python run_scripts/mt-bench/show_results.py --model $model --version $version
