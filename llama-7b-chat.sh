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

#module load anaconda/anaconda3
#export CONDA_ENVS=/data/others/devin/envs
#source activate /data/others/devin/envs/urial
#cd /nfsdata/data/devinh/URIAL

module load anaconda
module load cuda/cuda-11.1.0
module load cudnn/cudnn-8.0.4
export CONDA_ENVS=/nfsdata/data/devinh/envs
source activate $CONDA_ENVS/alignment

#filepath="result_dirs/alpaca_eval/aligned/Llama-2-7b-chat-hf_plain-instruct.json"
#gpu=0
#tps=1
#rp=1
#N=1
#urial="plain_instruct"
#batch=4
#CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
#    --urial $urial \
#    --engine hf \
#    --model_name "meta-llama/Llama-2-7b-chat-hf" \
#    --tensor_parallel_size $tps \
#    --dtype bfloat16 \
#    --data_name alpaca_eval --num_outputs $N \
#    --top_p 1.0 --temperature 0.0 --repetition_penalty $rp --batch_size $batch --max_tokens 2048 \
#    --hf_bf16 \
#    --filepath $filepath \
#    --overwrite

filepath="result_dirs/alpaca_eval/aligned/Llama-2-7b-chat-hf-vllm-subset.json"
gpu=0
tps=1
rp=1
N=1
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name alpaca_eval --num_outputs $N \
    --top_p 1.0 --temperature 0.7 --repetition_penalty $rp --batch_size 4 --max_tokens 2048 \
    --filepath $filepath \
    --subset_num 10 \
    --overwrite
