import json
import time
from retry.api import retry_call
import openai
import random
import keys
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
# from evaluate import load

THRESHOLD = 0.7

def read_json(file_path):
    # Open the JSON file
    with open(file_path, 'r') as file:
        # Load the JSON data into a dictionary
        data = json.load(file)
    # Now 'data' contains the dictionary representation of the JSON file
    return data

def get_response(prompt):
    chatgpt_query_time = time.time()
    # print(prompt)
    try:
        completion = retry_call(openai.ChatCompletion.create, fkwargs={"model":"gpt-3.5-turbo",
            "messages":[{"role": "user", "content": prompt}],
            "api_key":random.choice(keys.keys),
            "n":1,"temperature":0.0, "request_timeout":30}, tries=3, delay=1, jitter=1)
        #completion = openai.ChatCompletion.create(
        #    model="gpt-3.5-turbo",
        #    messages=[{"role": "user", "content": prompt}],
        #    api_key=random.choice(keys),
        #    n=4,
        #    temperature=0.25,
        #)
    except:
        print('-----------------------------openai API is failed!!!!!------------------------------------')
        completion = {'choices':[{'message':{'content':'Error'}}]}
    """
    nocontext_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": no_context_prompt}],
        api_key=random.choice(keys),
        n=3,
        temperature=0.25,
    )
    """
    print("chatgpt query time is : {}".format(str(time.time()-chatgpt_query_time)))
    return completion

def get_input(file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    list_ = []

    for (a,b,c) in zip(df['Level_1'], df['Level_2'], df['Level_3']):
        dict_ = {'Level_1':a, 'Level_2':b, 'Level_3':c}
        list_.append(dict_)
    return list_

def compare_references(refers, cands):
    if len(refers) != len(cands):
        print('ERROR! The length for references and candidates are not the same!')
    else:
        references, candidates = [], []
        correct_count = 0
        for ref, cand in zip(refers, cands):
            if ref.strip() == cand.strip():
                correct_count += 1
            # Tokenize sentences
            references.append(ref.split())
            candidates.append(cand.split())

        # Calculate BLEU for each pair
        scores = [sentence_bleu([ref], cand) for ref, cand in zip(references, candidates)]

        bleu_1_scores = [sentence_bleu([ref], cand, weights=(1, 0, 0, 0)) for ref, cand in zip(references, candidates)]
        bleu_2_scores = [sentence_bleu([ref], cand, weights=(0.5, 0.5, 0, 0)) for ref, cand in zip(references, candidates)]
        bleu_3_scores = [sentence_bleu([ref], cand, weights=(0.33, 0.33, 0.33, 0)) for ref, cand in zip(references, candidates)]
        bleu_4_scores = [sentence_bleu([ref], cand, weights=(0.25, 0.25, 0.25, 0.25)) for ref, cand in zip(references, candidates)]

        # Calculate average BLEU
        average_bleu = sum(scores) / len(scores)
        print(f"Average BLEU Score: {average_bleu}")
        correctness = float(correct_count / len(references))
        print(f"Perfect Match Rate: {correctness}")

        average_bleu1 = sum(bleu_1_scores) / len(scores)
        print(f"Average bleu_1 Score: {average_bleu1}")
        average_bleu2 = sum(bleu_2_scores) / len(scores)
        print(f"Average bleu_2 Score: {average_bleu2}")
        average_bleu3 = sum(bleu_3_scores) / len(scores)
        print(f"Average bleu_3 Score: {average_bleu3}")
        average_bleu4 = sum(bleu_4_scores) / len(scores)
        print(f"Average bleu_4 Score: {average_bleu4}")

        # mauve = load('mauve')
        # mauve_results = mauve.compute(predictions=cands, references=refers)
        # print(f"mauve: {mauve_results.mauve}")


def extract(content):
    lists = [x.strip() for x in content.strip().split('\n') if x.strip() != '']
    print(lists)

    level_2_result, level_1_result = "", ""
    for i, item in enumerate(lists):
        if '2. conversion' in item.lower() and i < len(lists)-1:
            level_2_result = lists[i+1]
        if 'further conversion' in item.lower() and i < len(lists)-1:
            level_1_result = lists[i+1]
    return level_2_result, level_1_result


def get_questions(turn1_path, turn2_path):
    path = 'run_scripts/mt-bench/question.reason_math.jsonl'
    # Initialize an empty list to store the dictionaries
    reference_data = []

    # Open and read the JSONL file
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object and append to the list
            reference_data.append(json.loads(line.strip()))

    turn1_data, turn2_data = read_json(turn1_path), read_json(turn2_path)

    return reference_data, turn1_data, turn2_data


def print_res(responses, turn_name, urial_name, model_id):
    # Step 1: Count the occurrences of each category
    count_yes = responses.count('YES')
    count_no = responses.count('NO')
    count_uncertain = responses.count('UNCERTAIN')

    # Step 2: Calculate the total number of elements in the list
    total_count = len(responses)

    # Step 3: Compute the percentage for each category
    percentage_yes = (count_yes / total_count) * 100
    percentage_no = (count_no / total_count) * 100
    percentage_uncertain = (count_uncertain / total_count) * 100

    # Display the results
    # print(responses)
    print(f"{model_id} {urial_name} {turn_name} Percentage of YES: {percentage_yes:.2f}%, NO: {percentage_no:.2f}%, UNCERTAIN: {percentage_uncertain:.2f}%")


def calc(reference_data, turn1_data, turn2_data, urial_name, model_id):
    res1, res2 = [], []
    # Now 'data' contains a list of dictionaries
    for entry in reference_data:
        # print(entry)
        question1, question2 = entry["turns"][0], entry["turns"][1]
        ref1, ref2 = entry["reference"][0], entry["reference"][1]
        pred1, pred2 = None, None
        for temp_dict in turn1_data:
            if temp_dict["question_id"] == entry["question_id"]:
                pred1 = temp_dict["turn1_output"]
        for temp_dict in turn2_data:
            if temp_dict["question_id"] == entry["question_id"]:
                pred2 = temp_dict["turn2_output"]

        if pred1 is None or pred2 is None:
            if pred1 is None:
                print('Turn 1 answer of the question {} is error'.format(entry["question_id"]))
            if pred2 is None:
                print('Turn 2 answer of the question {} is error'.format(entry["question_id"]))

        prompt = ""
        # Open the file in read mode
        with open('judgement_prompt.txt', 'r') as file:
            # Read all lines
            lines = file.readlines()
        for line in lines:
            prompt += line
        prompt = prompt.strip()

        temp_prompt = prompt.replace("$$$$$", question1).replace("&&&&&", ref1).replace("#####", pred1)
        completion = get_response(temp_prompt)
        content = completion['choices'][0]['message']['content']
        print(content)
        res1.append(content)

        temp_prompt = prompt.replace("$$$$$", question2).replace("&&&&&", ref2).replace("#####", pred2)
        completion = get_response(temp_prompt)
        content = completion['choices'][0]['message']['content']
        print(content)
        res2.append(content)

    print_res(res1, 'turn1', urial_name, model_id)
    print_res(res2, 'turn2', urial_name, model_id)
    print_res(res1+res2, 'all turns', urial_name, model_id)



if __name__ == "__main__":
    reference_data, turn1_data, turn2_data = get_questions('result_dirs/mt-bench/urial_value_impact_rewrite_upgrade_icl_8_15_1/Llama-2-7b-hf.turn1.json', 'result_dirs/mt-bench/urial_value_impact_rewrite_upgrade_icl_8_15_1/Llama-2-7b-hf.turn2.json')
    calc(reference_data, turn1_data, turn2_data, 'urial_value_impact_rewrite_upgrade_icl_8_15_1', 'Llama-2-7b-hf')

