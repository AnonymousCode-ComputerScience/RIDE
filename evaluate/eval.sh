run_name=$1 

if [ "${run_name}" == "urial-llama-70b" ]; then
    model_name="Llama-2-70b-urial"
    target_file="result_dirs/alpaca_eval/urial/llama-70b-urial.inst_help_v5-1k.json"
    ref_name="Llama-2-70b-chat-hf"
    ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"
elif [ "${run_name}" == "urial-llama-7b" ]; then
#    model_name="Llama-2-7b-valueimpact_vs_inst_1k_v4"
#    model_name="Llama-2-7b-chat-no-instruct_vs_7b-plain-instruct_norepeat"
#    model_name="Llama-2-7b-inst_1k_v4_vs_7b-chat-no-instruct"
#     model_name="Llama-2-7b-inst_1k_v4_icl3_vs_inst_1k_v4-vllm"
#    model_name="Llama-2-7b-valueimpact_vs_7b-chat-no-instruct"
#    model_name="Llama-2-7b-chat-plain-instruct_vs_7b-chat-no-instruct"
#    model_name="Llama-2-7b-inst_1k_v4_vs_7b-chat-plain-instruct-norepeat"
#    model_name="Llama-2-7b-valueimpact_vs_7b-inst_1k_v4-vllm"
#     model_name="Llama-2-7b_nollama_vllm_claude3_after_3_iteration_vs_7b-inst_1k_v4-vllm"
#     model_name="Llama-2-7b-urial.inst_help_v6-1k_vs_1k_v4-vllm-full"
#     model_name="Llama-2-7b-value_impact_rank16_rewrite_upgrade_vs_7b-inst_1k_v4-vllm-full"
     model_name="Llama-2-7b-value_impact_rewrite_upgrade_icl_16_8_sorry7-variant_vs_7b-inst_1k_v4-vllm-GPT4O-EVALUATOR"
#     model_name="Llama-2-7b_nollama_vllm_gpt4o_after_2_iteration_vs_7b-inst_1k_v4-vllm"
#    target_file="result_dirs/alpaca_eval/urial=value_impact_instruct/Llama-2-7b-hf-norepeat.json"
#    target_file="result_dirs/alpaca_eval/aligned/Llama-2-7b-chat-hf_no-system-prompt.json"
#    target_file="result_dirs/alpaca_eval/urial=inst_1k_v4/Llama-2-7b-hf-norepeat.json"
#    target_file="result_dirs/alpaca_eval/vllm_urial-inst_help_v5_1k/llama-7b-urial.inst_help_v5_1k.json"
#    target_file="result_dirs/alpaca_eval/vllm_urial-gpt4o_instruct_after_2_iteration/rp=1.1_N=1_T=0.3/Llama-2-7b-hf.json"
#    target_file="result_dirs/alpaca_eval/vllm-value_impact_rewrite_upgrade_icl_8_15_1.random_swap/rp=1.1_N=1_T=0.3/Llama-2-7b-hf.json"
     target_file="result_dirs/alpaca_eval/vllm-value_impact_rewrite_upgrade_icl_16_8_sorry7-variant/rp=1.1_N=1_T=0.3/Llama-2-7b-hf.json"
#    target_file="result_dirs/alpaca_eval/vllm_urial-inst_help_v6-1k/rp=1.1_N=1_T=0.3/Llama-2-7b-hf.json"
#    ref_name="Llama-2-7b-hf-norepeat"
#    ref_name="urial=inst_1k_v4"
#    ref_name="Llama-2-7b-chat-hf_no-system-prompt"
    ref_name="Llama-2-7b-inst_1k_v4-vllm"
#    ref_name="Llama-2-7b-chat-hf-vllm"
#    ref_name="value_impact_rewrite_upgrade_icl_8_15_1"
#    ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"
    ref_file="result_dirs/alpaca_eval/vllm_urial-inst_1k_v4/rp=1.1_N=1_T=0.3/Llama-2-7b-hf.json"
#    ref_file="result_dirs/alpaca_eval/vllm_urial-value_impact_rewrite_upgrade_icl_8_15_1/rp=1.1_N=1_T=0.3/Llama-2-7b-hf.json"
#     ref_file="result_dirs/alpaca_eval/vllm_urial-inst_1k_v4/rp=1.1_N=1_T=0.3/Llama-2-7b-hf-subset.json"
elif [ "${run_name}" == "urial-mistral" ]; then
    model_name="Mistral-7b-v0.1-value_impact_rewrite_upgrade_icl_8_15_1.random_swap-vs-icl_8_15_1-vllm-full-GPT4O-EVALUATOR"
    target_file="result_dirs/alpaca_eval/vllm-value_impact_rewrite_upgrade_icl_8_15_1.random_swap/rp=1.1_N=1_T=0.3/mistral-7b-v0.1.json"
    ref_name="vllm_mistral-value_impact_rewrite_upgrade_icl_8_15_1"
    ref_file="result_dirs/alpaca_eval/vllm_mistral-value_impact_rewrite_upgrade_icl_8_15_1/rp=1.1_N=1_T=0.3/mistral-7b-v0.1.json"
elif [ "${run_name}" == "urial-olmo" ]; then
    model_name="Olmo-7b-value_impact_rewrite_upgrade_icl_16_8_sorry.reduced-vs-inst_1k_v4-vllm-full-GPT4O-EVALUATOR"
    target_file="result_dirs/alpaca_eval/vllm-value_impact_rewrite_upgrade_icl_16_8_sorry.reduced/rp=1.15_N=1_T=0.0/olmo-7b.json"
    ref_name="vllm_olmo-7b-inst_1k_v4"
    ref_file="result_dirs/alpaca_eval/vllm_olmo-inst_1k_v4/rp=1.15_N=1_T=0.0/olmo-7b.json"
else
    echo "mode not supported"
    exit 1
fi

eval_folder="evaluate/results/ref=${ref_name}/"
mkdir -p $eval_folder

## evaluation for just_eval dataset
#if [ "${run_name}" == "urial-llama-70b" ]; then
#    model_name="Llama-2-70b-urial"
#    target_file="result_dirs/alpaca_eval/urial/llama-70b-urial.inst_help_v5-1k.json"
#    ref_name="Llama-2-70b-chat-hf"
#    ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"
#elif [ "${run_name}" == "urial-llama-7b" ]; then
#    model_name="Llama-2-7b-value_impact_rewrite_upgrade_icl_16_8_10_vs_URIAL_inst_1k_v4-vllm-full-GPT4O-EVALUATOR"
#    target_file="result_dirs/just_eval/vllm_urial-value_impact_rewrite_upgrade_icl_16_8_10/rp=1.1_N=1_T=0.3/Llama-2-7b-hf.json"
#    ref_name="urial=inst_1k_v4"
#    ref_file="result_dirs/just_eval/vllm_urial-inst_1k_v4/rp=1.1_N=1_T=0.3/Llama-2-7b-hf.json"
#elif [ "${run_name}" == "urial-mistral" ]; then
#    model_name="Mistral-7b-v0.1-value_impact_rewrite_upgrade_icl_8_15_1.reduced-vs_rewrite_upgrade_icl_8_15_1-vllm-full-GPT4O-EVALUATOR"
#    target_file="result_dirs/just_eval/vllm_mistral-value_impact_rewrite_upgrade_icl_8_15_1/rp=1.1_N=1_T=0.3/mistral-7b-v0.1.json"
#    ref_name="Mistral_inst_1k_v4"
#    ref_file="result_dirs/just_eval/vllm_mistral-inst_1k_v4/rp=1.1_N=1_T=0.3/mistral-7b-v0.1.json"
#elif [ "${run_name}" == "urial-olmo" ]; then
#    model_name="Olmo-7b-value_impact_rewrite_upgrade_icl_16_8_10.reduced-vs-inst_1k_v4-vllm-full-GPT4O-EVALUATOR"
#    target_file="result_dirs/just_eval/vllm_olmo-value_impact_rewrite_upgrade_icl_16_8_10.reduced/rp=1.15_N=1_T=0.0/olmo-7b.json"
#    ref_name="vllm_olmo-7b-inst_1k_v4"
#    ref_file="result_dirs/just_eval/vllm_olmo-inst_1k_v4/rp=1.15_N=1_T=0.0/olmo-7b.json"
#else
#    echo "mode not supported"
#    exit 1
#fi
#
#
#eval_folder="evaluate/results/just_eval/ref=${ref_name}/"
#mkdir -p $eval_folder


n_shards=8
#shard_size=13
shard_size=101
#shard_size=126
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    eval_file="${eval_folder}/${model_name}.$start-$end.json"
    python evaluate/eval.py \
        --model gpt-4o \
        --action eval \
        --mode pairwise \
        --eval_template evaluate/eval_template_pairwise.md \
        --model_output_file $target_file \
        --ref_output_file $ref_file \
        --eval_output_file $eval_file \
        --api_key ** \
        --start_idx $start --end_idx $end  &
done
# --model gpt-4o \

# Wait for all background processes to finish
wait

# Run the merge results script after all evaluation scripts have completed
python evaluate/merge_results.py $eval_folder $model_name