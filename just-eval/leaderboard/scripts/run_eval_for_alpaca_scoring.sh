model_name=$1 
target_file="leaderboard/outputs/alpaca_eval_for_scoring/${model_name}.json"
eval_parent_folder="leaderboard/eval_results/gpt-4o/alpaca_eval_for_scoring/"
eval_folder="${eval_parent_folder}/${model_name}/"
mkdir -p $eval_folder

start_gpu=0 # not useful, just a placeholder

## ---------------------- FOR FULL TEST DATASET ----------------------
n_shards=8
shard_size=101
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    eval_file="${eval_folder}/${model_name}.general.$start-$end.json"
    just_eval \
        --mode score_multi \
        --gpt_model gpt-4o-2024-05-13 \
        --model_output_file $target_file \
        --eval_output_file $eval_file \
        --start_idx $start --end_idx $end  &
done

n_shards=8
shard_size=101
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    eval_file="${eval_folder}/${model_name}.safety.$start-$end.json"
    just_eval \
        --mode score_safety \
        --gpt_model gpt-4o-2024-05-13 \
        --model_output_file $target_file \
        --eval_output_file $eval_file \
        --start_idx $start --end_idx $end  &
done

## ---------------------- FOR SUBSET ----------------------
#n_shards=3
#shard_size=9
#for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
#    eval_file="${eval_folder}/${model_name}.safety.$start-$end.json"
#    just_eval \
#        --mode score_safety \
#        --gpt_model gpt-4o-2024-05-13 \
#        --model_output_file $target_file \
#        --eval_output_file $eval_file \
#        --start_idx $start --end_idx $end  &
#done
#
#n_shards=7
#shard_size=11
#for ((start = 27, end = ((27 + $shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
#    eval_file="${eval_folder}/${model_name}.general.$start-$end.json"
#    just_eval \
#        --mode score_multi \
#        --gpt_model gpt-4o-2024-05-13 \
#        --model_output_file $target_file \
#        --eval_output_file $eval_file \
#        --start_idx $start --end_idx $end  &
#done


# Wait for all background processes to finish
wait
echo "All evaluation scripts have completed"
# Run the merge results script after all evaluation scripts have completed
python leaderboard/scripts/merge_results.py $eval_folder $model_name.general
python leaderboard/scripts/merge_results.py $eval_folder $model_name.safety
mv $eval_folder/$model_name.general.json $eval_parent_folder/$model_name.score_multi.json 
mv $eval_folder/$model_name.safety.json $eval_parent_folder/$model_name.score_safety.json