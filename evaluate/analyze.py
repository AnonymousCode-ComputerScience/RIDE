import argparse
import os
import json
import openai 
import random
from pathlib import Path
from itertools import combinations
from string import Template
from tqdm import tqdm
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor
from eval_utils import (
    retry_handler, 
    openai_chat_request, 
)
from datasets import load_dataset

def read_json(file_path):
    # Open the JSON file
    with open(file_path, 'r') as file:
        # Load the JSON data into a dictionary
        data = json.load(file)
    # Now 'data' contains the dictionary representation of the JSON file
    return data


def write_json(data, savedir):
    with open(savedir, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def analyze(comparison_file='results/ref=urial=inst_1k_v4/Llama-2-7b-valueimpact_vs_inst_1k_v4.json', metric='helpfulness',
            wl=None, target='value_impact_instruct', refer='inst_1k_v4'):
    if wl is None:
        wl = ['win', 'lose', 'tie']
    win_, lose_, tie_ = [], [], []
    folder_path = os.path.dirname(comparison_file)
    comparison = read_json(comparison_file)
    metrics = {metric: {target: 0, refer: 0, "tie": 0}}
    for item_ in comparison:
        if 'parsed_result' in item_:
            dict_ = {'input': item_['input'],
                     'model_output': item_['model_output'],
                     'ref_output': item_['ref_output'],
                     'generator': item_['generator'],
                     'ref_generator': item_['ref_generator'],
                     'assignment': item_['assignment'],
                     'parsed_result': item_['parsed_result']}
            for key, value in metrics.items():
                if item_["parsed_result"]["choices"][key] in ['A', 'B']:
                    if refer in item_["assignment"][item_["parsed_result"]["choices"][key]]:
                        value[refer] += 1
                        lose_.append(dict_)
                    else:
                        value[target] += 1
                        win_.append(dict_)
                elif item_["parsed_result"]["choices"][key] == 'tie':
                    value["tie"] += 1
                    tie_.append(dict_)
                else:
                    print('error for item: {}'.format(item_))
    res_file_path = comparison_file.split('/')[-1].split('.')[0]
    if 'win' in wl:
        path = folder_path + '/' + res_file_path + '_{}_win.json'.format(metric)
        write_json(win_, path)
    if 'lose' in wl:
        path = folder_path + '/' + res_file_path + '_{}_lose.json'.format(metric)
        write_json(lose_, path)
    if 'tie' in wl:
        path = folder_path + '/' + res_file_path + '_{}_tie.json'.format(metric)
        write_json(tie_, path)


def main():
    comparison_file = 'results/ref=urial=inst_1k_v4/Llama-2-7b-valueimpact_vs_inst_1k_v4.json'
    comparison = read_json(comparison_file)
    metrics = {"helpfulness": {"value_impact_instruct": 0, "URIAL-inst_1k_v4": 0, "tie": 0},
               "factuality": {"value_impact_instruct": 0, "URIAL-inst_1k_v4": 0, "tie": 0},
               "depth": {"value_impact_instruct": 0, "URIAL-inst_1k_v4": 0, "tie": 0},
               "engagement": {"value_impact_instruct": 0, "URIAL-inst_1k_v4": 0, "tie": 0},
               "clarity": {"value_impact_instruct": 0, "URIAL-inst_1k_v4": 0, "tie": 0},
               "safety": {"value_impact_instruct": 0, "URIAL-inst_1k_v4": 0, "tie": 0}}
    for item_ in comparison:
        if 'parsed_result' in item_:
            for key, value in metrics.items():
                if item_["parsed_result"]["choices"][key] in ['A', 'B']:
                    if 'inst_1k_v4' in item_["assignment"][item_["parsed_result"]["choices"][key]]:
                        value["URIAL-inst_1k_v4"] += 1
                    else:
                        value["value_impact_instruct"] += 1
                elif item_["parsed_result"]["choices"][key] == 'tie':
                    value["tie"] += 1
                else:
                    print('error for item: {}'.format(item_))
    for key, value in metrics.items():
        sum_ = value["value_impact_instruct"] + value["URIAL-inst_1k_v4"] + value["tie"]
        str_ = key + ': ' + \
               'value_impact_instruct win: ' + str(value["value_impact_instruct"]) + '/' + str(sum_) + ': ' + str(float(value["value_impact_instruct"]/sum_)) \
               + 'tie: ' + str(value["tie"]) + '/' + str(sum_) + ': ' + str(float(value["tie"]/sum_)) \
               + 'loss: ' + str(value["URIAL-inst_1k_v4"]) + '/' + str(sum_) + ': ' + str(float(value["URIAL-inst_1k_v4"]/sum_))
        print(str_)

def get_key_words(metric, file_path):
    res_ = set()
    if metric == 'engagement':
        key_words = ['engag']
    elif metric == 'depth':
        key_words = ['depth', 'deep']
    elif metric == 'helpfulness':
        key_words = ['help', 'useful']
    elif metric == 'factuality':
        key_words = ['fact']
    elif metric == 'clarity':
        key_words = ['clarity', 'clear']
    elif metric == 'safety':
        key_words = ['safe', 'harm']
    else:
        key_words = ['good', 'better']
    data_ = read_json(file_path)
    for item_ in data_:
        if 'parsed_result' in item_:
            if 'rationale' in item_['parsed_result']:
                rationale = item_['parsed_result']['rationale']
                rationales = [x.strip().lower() for x in rationale.split('.')]
                for key_word in key_words:
                    for rationale in rationales:
                        if key_word in rationale:
                            res_.add(rationale)
    res_path = os.path.dirname(file_path) + '/' + '{}_rationales.json'.format(file_path.split('/')[-1].split('.')[0])
    write_json(list(res_), res_path)


if __name__ == "__main__":
    # main()
    analyze(metric='safety')
    get_key_words(metric='safety', file_path='results/ref=urial=inst_1k_v4/Llama-2-7b-valueimpact_vs_inst_1k_v4_safety_lose.json')
