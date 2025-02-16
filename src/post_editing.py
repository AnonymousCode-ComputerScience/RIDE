import argparse
from tqdm import tqdm
import json
import os
PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(PROJECT_ABSOLUTE_PATH)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urial', default='vllm-value_impact_rewrite_upgrade_icl_16_8_sorry7.reduced', type=str)
    parser.add_argument('--filepath', default="result_dirs/alpaca_eval/vllm-value_impact_rewrite_upgrade_icl_16_8_sorry7.reduced/rp=1.15_N=1_T=0.0/olmo-7b.json", type=str)
    parser.add_argument('--data_name', default="alpaca_eval", type=str)
    parser.add_argument('--mode', default="score", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filepath = PROJECT_ABSOLUTE_PATH + '/' + args.filepath

    with open(filepath, 'r') as file:
        data = json.load(file)

    if args.data_name == "alpaca_eval" and args.mode == 'score':
        res_ = []
        for index_, item_ in enumerate(data):
            temp_ = {"id": index_, "instruction": item_["instruction"],
                     "output": item_["output"][0], "generator": item_["generator"],
                     "dataset": item_["dataset"], "model_input": item_["model_input"]}
            res_.append(temp_)
        with open(filepath.replace('.json', '_for-score.json'), 'w') as json_file:
            json.dump(res_, json_file, indent=4)

    else:
        if '_COT' in args.urial:
            if args.data_name == 'just_eval':
                for item_ in data:
                    res_ = item_['output']
                    if '\n\nResponse:' in res_:
                        res_ = res_.split('\n\nResponse:')[1].strip()
                    else:
                        print('sth wrong in {}'.format(res_))
                    item_['output'] = res_
            elif args.data_name == "alpaca_eval":
                for item_ in data:
                    res_ = []
                    for o in item_['output']:
                        if '\n\nResponse:' in o:
                            res_.append(o.split('\n\nResponse:')[1].strip())
                        else:
                            print('sth wrong in {}'.format(o))
                    item_['output'] = res_

        elif args.data_name == 'just_eval' and 'subset.json' in args.filepath:
            multi_, safety_ = [], []
            for item_ in data:
                if item_["dataset"] in ['MaliciousInstruct', 'hh-rlhf/red-team-attempts']:
                    safety_.append(item_)
                else:
                    multi_.append(item_)
            print('multi_ length: {}, safety_ length: {}'.format(len(multi_), len(safety_)))
            data = safety_ + multi_

        with open(filepath.replace('Llama-2-7b-hf', 'Llama-2-7b-hf-postedit'), 'w') as json_file:
            json.dump(data, json_file, indent=4)



