version=$1
temp=${2:-0}
rp=${3:-1}
output_dir="result_dirs/mt-bench/urial_bench/"
mkdir -p $output_dir
gpus=${4:-"0"}
tsp=1

n_shards=4
shard_size=20
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do

    CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
        --start_index $start --end_index $end \
        --engine hf \
        --urial $version \
        --download_dir /net/nfs/s2-research/llama2/ \
        --model_name microsoft/phi-2 \
        --dtype bfloat16 \
        --data_name mt-bench \
        --mt_turn 1 \
        --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 1 --max_tokens 2048 \
        --filepath $output_dir/phi-2.turn1.${start}-${end}.json \
        --overwrite &
done 
wait 
python evaluate/merge_results.py $output_dir/ phi-2.turn1


n_shards=4
shard_size=20
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
        --start_index $start --end_index $end \
        --engine hf \
        --urial $version \
        --download_dir /net/nfs/s2-research/llama2/ \
        --model_name microsoft/phi-2 \
        --dtype bfloat16 \
        --data_name mt-bench \
        --mt_turn 2 \
        --mt_turn1_result $output_dir/phi-2.turn1.json \
        --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 1 --max_tokens 2048 \
        --filepath $output_dir/phi-2.turn2.${start}-${end}.json \
        --overwrite &
done 
wait 
python evaluate/merge_results.py $output_dir/ phi-2.turn2