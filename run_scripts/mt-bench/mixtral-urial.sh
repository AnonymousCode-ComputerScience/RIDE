version=$1
temp=${2:-0}
rp=${3:-1}
output_dir="result_dirs/mt-bench/urial_bench/"
mkdir -p $output_dir
gpu=0,1,2,3
n=4
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --tensor_parallel_size $n \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 1 --max_tokens 2048 \
    --filepath $output_dir/Mixtral-8x7B-v0.1.turn1.json \
    --overwrite 


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --tensor_parallel_size $n \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/Mixtral-8x7B-v0.1.turn1.json \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 1 --max_tokens 2048 \
    --filepath $output_dir/Mixtral-8x7B-v0.1.turn2.json \
    --overwrite 