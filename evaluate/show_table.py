import json 
from datasets import load_dataset
from tabulate import tabulate


# configs = [
#         {
#             "name": "mistral-urial",
#             "filepath": "evaluate/results/ref=Mistral-7B-Instruct-v0.1/mistral-urial.json",
#             'ref': "Mistral-7B-Instruct-v0.1"
#         },
#         {
#             "name": "Llama-2-7b-urial",
#             "filepath": "evaluate/results/ref=Llama-2-7b-chat-hf/Llama-2-7b-urial-valueimpact_vs_aligned.json",
#             'ref': "Llama-2-7b-chat-hf"
#         },
#         {
#             "name": "Llama-2-70b-urial",
#             "filepath": "evaluate/results/ref=Llama-2-70b-chat-hf/Llama-2-70b-urial.json",
#             'ref': "Llama-2-70b-chat-hf"
#         },
#     ]

# # 'ref' should refer to the 'assignment' in evaluate json.
# configs = [
#         {
#             "name": "Llama-2-7b-valueimpact-norepeat",
#             "filepath": "/nfsdata/data/devinh/URIAL/evaluate/results/ref=Llama-2-7b-chat-hf_no-system-prompt/Llama-2-7b-valueimpact_vs_7b-chat-no-instruct.json",
#             'ref': "meta-llama/Llama-2-7b-chat-hf-no_instruct"
#         }
#     ]

# configs = [
#         {
#             "name": "Llama-2-7b-value_impact_rewrite_upgrade_icl_8_15_1.random_swap",
#             "filepath": "/Users/dhua0029/PycharmProjects/URIAL/evaluate/results/ref=value_impact_rewrite_upgrade_icl_8_15_1/Llama-2-7b-value_impact_rewrite_upgrade_icl_8_15_1.random_swap_vs_icl_8_15_1-vllm-full-GPT4O-EVALUATOR.json",
#             'ref': "meta-llama/Llama-2-7b-hf-value_impact_rewrite_upgerade_icl_8_15_1"
#         }
#     ]
#
# configs = [
#         {
#             "name": "Mistral-7b-v0.1-value_impact_rewrite_upgrade_icl_8_15_1.random_swap",
#             "filepath": "/Users/dhua0029/PycharmProjects/URIAL/evaluate/results/ref=vllm_mistral-value_impact_rewrite_upgrade_icl_8_15_1/Mistral-7b-v0.1-value_impact_rewrite_upgrade_icl_8_15_1.random_swap-vs-icl_8_15_1-vllm-full-GPT4O-EVALUATOR.json",
#             'ref': "mistralai/Mistral-7b-v0.1-value_impact_rewrite_upgrade_icl_8_15_1"
#         }
#     ]

configs = [
        {
            "name": "Llama-2-7b-value_impact_rewrite_upgrade_icl_16_8_sorry7-variant",
            "filepath": "/Users/dhua0029/PycharmProjects/URIAL/evaluate/results/ref=Llama-2-7b-inst_1k_v4-vllm/Llama-2-7b-value_impact_rewrite_upgrade_icl_16_8_sorry7-variant_vs_7b-inst_1k_v4-vllm-GPT4O-EVALUATOR.json",
            'ref': "meta-llama/Llama-2-7b-hf-inst_1k_v4"
        }
    ]

# configs = [
#         {
#             "name": "Llama-2-7b-chat-plain-instruct_norepeat",
#             "filepath": "/Users/dhua0029/PycharmProjects/URIAL/evaluate/results/ref=Llama-2-7b-hf-norepeat/Llama-2-7b-chat_vs_7b-plain-instruct_norepeat.json",
#             'ref': "meta-llama/Llama-2-7b-hf-plain_instruct"
#         }
#     ]

# configs = [
#         {
#             "name": "Olmo-7b-value_impact_rewrite_upgrade_icl_16_8_10",
#             "filepath": "/Users/dhua0029/PycharmProjects/URIAL/evaluate/results/just_eval/ref=vllm_olmo-7b-inst_1k_v4/Olmo-7b-value_impact_rewrite_upgrade_icl_16_8_10.reduced-vs-inst_1k_v4-vllm-full-GPT4O-EVALUATOR.json",
#             'ref': "allenai/OLMo-7B-inst_1k_v4"
#         }
#     ]


aspects = ["helpfulness", "factuality", "depth", "engagement", "clarity", "safety"]
for config in configs:
    win_rows = []
    tie_rows = []
    lose_rows = []

    with open(config["filepath"]) as f:
        data = json.load(f)
    print('the file is {}'.format(config["filepath"]))
    tie_counts = {x:0 for x in aspects}
    win_counts = {x:0 for x in aspects}
    lose_counts = {x:0 for x in aspects}
    model_output_lens = []
    ref_output_lens = []
    for item in data:
        if item["assignment"]["A"] == config["ref"]:
            model_assignment = "B"
            ref_assignment = "A"
        else:
            model_assignment = "A"
            ref_assignment = "B"
        if "parsed_result" not in item:
            continue
        for aspect in aspects:
            c = item["parsed_result"]["choices"][aspect]
            if c == "tie":
                tie_counts[aspect] += 1
            elif c == model_assignment:
                win_counts[aspect] += 1
            elif c == ref_assignment:
                lose_counts[aspect] += 1
        
        mo_len = len(item["model_output"]) if type(item["model_output"]) == str else len(item["model_output"][0])
        ref_len = len(item["ref_output"]) if type(item["ref_output"]) == str else len(item["ref_output"][0])
        model_output_lens.append(mo_len)
        ref_output_lens.append(ref_len)
    total_dict = {x:tie_counts[x]+win_counts[x]+lose_counts[x] for x in aspects}
    
    # print(json.dumps(total_dict, indent=2))

    tie_rates = {x:tie_counts[x]/total_dict[x] for x in aspects}
    win_rates = {x:win_counts[x]/total_dict[x] for x in aspects}
    lose_rates = {x:lose_counts[x]/total_dict[x] for x in aspects}
    
    win_row = [f"{config['name']} Win:"]
    tie_row = [f"{config['name']} Tie:"]
    lose_row = [f"{config['name']} Lose:"]
    for aspect in aspects:
        win_row.append((win_rates[aspect])*100)
        tie_row.append((tie_rates[aspect])*100)
        lose_row.append((lose_rates[aspect])*100)

    win_rows.append(win_row)
    tie_rows.append(tie_row)
    lose_rows.append(lose_row)

    tables = win_rows + tie_rows + lose_rows
    print("-"*120 + "\n")
    avg_model_len = sum(model_output_lens)/len(model_output_lens)
    avg_ref_len = sum(ref_output_lens)/len(ref_output_lens)
    title = f"{config['name']} ({avg_model_len:.1f}) VS {config['ref']} ({avg_ref_len:.1f})"
    print(f"\t\t\t\t{title}")    
    print(tabulate(tables, headers=["model",]+aspects, tablefmt="github", floatfmt=".2f"))

 
    # print(f"""
    # aspects = {aspects}
    # title = '{title}'
    # wins = {win_row[1:]}
    # ties = {tie_row[1:]}
    # loses = {lose_row[1:]}
    # """)


# print(tabulate(win_rows, headers=["model",]+aspects, tablefmt="github", floatfmt=".2f"))
# print()
# print("="*100)
# print()
# print(tabulate(win_tie_rows, headers=["model",]+aspects, tablefmt="github", floatfmt=".2f"))
