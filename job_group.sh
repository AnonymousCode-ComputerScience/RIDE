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
#SBATCH --mail-user=devin.hua@monash.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --qos=conf

#module load anaconda/anaconda3
#export CONDA_ENVS=/data/others/devin/envs
#source activate /data/others/devin/envs/urial
#cd /nfsdata/data/devinh/URIAL
#
##urial="value_impact_instruct"
#urial="inst_1k_v4"
#output_dir="result_dirs/alpaca_eval/urial=${urial}/"
#filepath="result_dirs/alpaca_eval/urial=${urial}/Llama-2-7b-hf-norepeat.json"

#CUDA_VISIBLE_DEVICES=0 python src/unified_infer.py \
#    --urial $urial \
#    --engine hf \
#    --model_name "meta-llama/Llama-2-7b-hf" \
#    --tensor_parallel_size 1 \
#    --dtype bfloat16 \
#    --data_name "alpaca_eval" \
#    --top_p 1.0 \
#    --temperature 0.3 \
#    --repetition_penalty 1.1 \
#    --batch_size 16 \
#    --max_tokens 2048 \
#    --hf_bf16 \
#    --filepath $filepath \
#    --overwrite \
#    --output_folder $output_dir/

# If batch size if bigger than one, in a batch, the decoded examples that generates the stop words will not early stop.
# Refer to EndOfFunctionCriteria() in hf_models.py.
# no_repeat_ngram_size would prohibit the token which lead to repeat_ngram, it will not teminate the generation directly. But when it prohibit some tokens, the end_id may become the most possible candidate.

#CUDA_VISIBLE_DEVICES=0 python src/unified_infer.py \
#    --urial $urial \
#    --engine hf \
#    --model_name "meta-llama/Llama-2-7b-hf" \
#    --tensor_parallel_size 1 \
#    --dtype bfloat16 \
#    --data_name "alpaca_eval" \
#    --top_p 1.0 \
#    --temperature 0.3 \
#    --repetition_penalty 1.1 \
#    --no_repeat_ngram_size 5 \
#    --batch_size 1 \
#    --max_tokens 2048 \
#    --hf_bf16 \
#    --filepath $filepath \
#    --overwrite \
#    --output_folder $output_dir/

module load anaconda/anaconda3
module load cuda/cuda-11.1.0
module load cudnn/cudnn-8.0.4
export CONDA_ENVS=/nfsdata/data/devinh/envs
source activate $CONDA_ENVS/alignment
cd /nfsdata/data/devinh/URIAL

#version="inst_1k_v4_icl1"
#version="inst_1k_v4.reduced"
#version="value_impact_instruct"
#version="value_impact_no_llama_instruct_top3_rank"
#version="value_impact_no_llama_instruct_bfs"
#version="inst_help_v6-1k"
#version="claude3_opus_instruct_after_3_iteration"
#version="gpt4o_instruct_after_2_iteration"
#version="value_impact_no_llama_instruct_rank10_rewrite_upgrade"
#version="value_impact_rewrite_upgrade_icl_8_15_1"
#version="value_impact_rewrite_upgrade_icl_8_15_1.random_swap"
#version="value_impact_rewrite_upgrade_icl_8_15_1.reduced"
#version="value_impact_rewrite_upgrade_icl_16_8_10.reduced"
#version="value_impact_rewrite_twoturns_icl_16_8_10"
#version="value_impact_rewrite_twoturns_icl_8_15_1"
#version="value_impact_rewrite_upgrade_icl_8_15_1.no_order"
#version="value_impact_rewrite_upgrade_icl_16_8_10.no_order"
#version="value_impact_rewrite_upgrade_icl_16_8_safe"
#version="value_impact_rewrite_upgrade_icl_16_8_safe.reduced"
#version="value_impact_rewrite_upgrade_icl_16_8_sorry"
#version="value_impact_rewrite_upgrade_icl_16_8_sorry.reduced"
#version="value_impact_rewrite_upgrade_icl_16_8_10.three_steps"
#version="value_impact_rewrite_upgrade_icl_16_8_10.human"
#version="value_impact_rewrite_upgrade_icl_16_8_10.lengthy"
#version="value_impact_rewrite_upgrade_icl_16_8_10.lengthy.reduced"
#version="value_impact_rewrite_upgrade_icl_16_8_sorry_COT"
#version="value_impact_no_llama_instruct_rank15_rewrite_upgrade"
#version="value_impact_rewrite_upgrade_icl_16_8_sorry7-variant.reduced"
#version="sorry_urial"
version="no_instruct"

rp=1.1
N=1
T=0.3
output_dir="result_dirs/alpaca_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/"
filepath="result_dirs/alpaca_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/Llama-2-7b-hf.json"
mkdir -p $output_dir
tps=1
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name alpaca_eval --num_outputs $N \
    --top_p 1.0 --temperature $T --repetition_penalty $rp --batch_size 4 --max_tokens 2048\
    --filepath $filepath \
    --output_folder $output_dir/ \
    --overwrite

#--model_name meta-llama/Llama-2-7b-hf \

#python src/post_editing.py \
#    --urial $version \
#    --filepath $filepath \
#    --data_name alpaca_eval \

rp=1.1
N=1
T=0.3
output_dir="result_dirs/alpaca_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/"
filepath="result_dirs/alpaca_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/mistral-7b-instruct-v0.1.json"
mkdir -p $output_dir
tps=1
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --engine vllm \
    --model_name "mistralai/Mistral-7B-Instruct-v0.1" \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name alpaca_eval --num_outputs $N \
    --top_p 1.0 --temperature $T --repetition_penalty $rp --batch_size 4 \
    --filepath $filepath \
    --output_folder $output_dir/ \
    --overwrite

#    --model_name "mistralai/Mistral-7b-v0.1"



python src/post_editing.py \
    --urial $version \
    --filepath $filepath \
    --data_name alpaca_eval \
    --mode "score"


rp=1.15
N=1
T=0.0
output_dir="result_dirs/alpaca_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/"
filepath="result_dirs/alpaca_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/olmo-7b.json"
CACHE_DIR="tmp_cache/"
mkdir -p $CACHE_DIR
mkdir -p $output_dir
gpu=0
tps=1
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --engine vllm \
    --download_dir $CACHE_DIR \
    --model_name "allenai/OLMo-7B-Instruct-hf" \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name alpaca_eval --num_outputs $N \
    --top_p 1.0 --temperature $T --repetition_penalty $rp --batch_size 4 --max_tokens 2048 \
    --filepath $filepath \
    --output_folder $output_dir/ \
    --overwrite

# --subset_num 100 \
# --model_name "allenai/OLMo-7B" \


rp=1.1
N=1
T=0.3
output_dir="result_dirs/just_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/"
filepath="result_dirs/just_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/Llama-2-7b-hf.json"
mkdir -p $output_dir
tps=1
CACHE_DIR="tmp_cache/"
mkdir -p $CACHE_DIR
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir $CACHE_DIR \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name just_eval --num_outputs $N \
    --top_p 1.0 --temperature $T --repetition_penalty $rp --batch_size 4 --max_tokens 2048\
    --filepath $filepath \
    --output_folder $output_dir/ \
    --overwrite

#        --model_name meta-llama/Llama-2-7b-hf \


# --subset_num 100 \

#python src/post_editing.py \
#    --urial $version \
#    --filepath $filepath \
#    --data_name just_eval \

rp=1.1
N=1
T=0.3
output_dir="result_dirs/just_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/"
filepath="result_dirs/just_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/mistral-7b-v0.1.json"
mkdir -p $output_dir
CACHE_DIR="tmp_cache/"
mkdir -p $CACHE_DIR
tps=1
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --engine vllm \
    --model_name "mistralai/Mistral-7B-Instruct-v0.1" \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name just_eval --num_outputs $N \
    --top_p 1.0 --temperature $T --repetition_penalty $rp --batch_size 4 \
    --filepath $filepath \
    --output_folder $output_dir/ \
    --overwrite

#--model_name "mistralai/Mistral-7b-v0.1" \


rp=1.15
N=1
T=0.0
output_dir="result_dirs/just_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/"
filepath="result_dirs/just_eval/vllm-${version}/rp=${rp}_N=${N}_T=${T}/olmo-7b.json"
CACHE_DIR="tmp_cache/"
mkdir -p $CACHE_DIR
mkdir -p $output_dir
tps=1
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --engine vllm \
    --download_dir $CACHE_DIR \
    --model_name "allenai/OLMo-7B-Instruct-hf" \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name just_eval --num_outputs $N \
    --top_p 1.0 --temperature $T --repetition_penalty $rp --batch_size 4 --max_tokens 2048 \
    --filepath $filepath \
    --output_folder $output_dir/ \
    --overwrite

# --model_name "allenai/OLMo-7B" \