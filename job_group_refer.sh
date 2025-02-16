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

module load anaconda/anaconda3
export CONDA_ENVS=/data/others/devin/envs
source activate /data/others/devin/envs/urial
cd /nfsdata/data/devinh/URIAL

urial="plain_instruct"
output_dir="result_dirs/alpaca_eval/aligned/"
#filepath="result_dirs/alpaca_eval/aligned/Llama-2-7b-chat-hf-norepeat.json"

#CUDA_VISIBLE_DEVICES=0 python src/unified_infer.py \
#    --urial $urial \
#    --engine hf \
#    --model_name "meta-llama/Llama-2-7b-chat-hf" \
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

#filepath="result_dirs/alpaca_eval/aligned/Llama-2-7b-hf-norepeat.json"
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

filepath="result_dirs/alpaca_eval/aligned/Llama-2-7b-chat-hf-norepeat.json"
CUDA_VISIBLE_DEVICES=0 python src/unified_infer.py \
    --urial $urial \
    --engine hf \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --tensor_parallel_size 1 \
    --dtype bfloat16 \
    --data_name "alpaca_eval" \
    --top_p 1.0 \
    --temperature 0.3 \
    --repetition_penalty 1.1 \
    --no_repeat_ngram_size 5 \
    --batch_size 1 \
    --max_tokens 2048 \
    --hf_bf16 \
    --filepath $filepath \
    --overwrite \
    --output_folder $output_dir/






