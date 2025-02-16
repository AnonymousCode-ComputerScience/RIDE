# # stop_words = ["# Query", "# User"]
# # print("|".join(stop_words))
#
# from transformers import AutoTokenizer
# # hf_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
# hf_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
# test = hf_tokenizer.apply_chat_template(
#     [
#             {
#                 "role": "system",
#                 "content": "This is a test sentence.",
#             },
#             {
#                 "role": "user",
#                 "content": "This is a response.",
#             }
#     ]
#     , add_generation_prompt=True
#     # , tokenize=False
# )
# print(test)
import os

from datasets import load_dataset
import json

# # Load dataset
# dataset = load_dataset("sorry-bench/sorry-bench-202406", split='train')
#
# # Convert dataset to Python dictionary
# dataset_dict = dataset.to_dict()
#
# # Define path to save JSON file
# save_path = 'sorry-bench-202406.json'
#
# # Save as JSON
# with open(save_path, 'w', encoding='utf-8') as f:
#     json.dump(dataset_dict, f, ensure_ascii=False, indent=2)
#
# print(f'Dataset saved as {save_path}')

# dict_ = {}
#
# with open('sorry-bench-202406.json') as f:
#     formatted_outputs = json.load(f)
#     for item in zip(formatted_outputs["question_id"], formatted_outputs["category"], formatted_outputs["turns"], formatted_outputs["prompt_style"]):
#         if item[0] not in dict_:
#             dict_[item[0]] = []
#         dict_[item[0]].append({'turns': item[2], 'category': item[1], 'prompt_style': item[3]})
#
# save_path = 'sorry-bench-202406-organized.json'
# with open(save_path, 'w', encoding='utf-8') as f:
#     json.dump(dict_, f, ensure_ascii=False, indent=2)


# with open('sorry-bench-202406-organized.json') as f:
#     out_dict = json.load(f)
#
# with open('sorry-bench-human-judgment-202406.json') as f:
#     in_dict = json.load(f)
#
# for item in zip(in_dict["question_id"], in_dict["model_id"], in_dict["choices"], in_dict["prompt_style"], in_dict["human_score"]):
#     if str(item[0]) in out_dict:
#         for dict_ in out_dict[str(item[0])]:
#             if dict_["prompt_style"] == item[3]:
#                 if 'response' not in dict_:
#                     dict_["response"] = []
#                 dict_["response"].append(
#                     {'model_id': item[1],
#                      'choices': item[2],
#                      'human_score': item[4]})
#     else:
#         print('{} is not in {}'.format(item[0], 'sorry-bench-202406-organized.json'))
# save_path = 'sorry-bench-202406-organized-with-answer.json'
# with open(save_path, 'w', encoding='utf-8') as f:
#     json.dump(out_dict, f, ensure_ascii=False, indent=2)
#
# res_list = []
# for key, value in out_dict.items():
#     for dict_ in value:
#         if dict_["prompt_style"] == 'base' and 'response' in dict_:
#             question = dict_['turns'][0]
#             for res_ in dict_['response']:
#                 if res_["human_score"] == 0.0:
#                     response = res_["choices"][0]['turns'][0]
#                     model_id = res_["model_id"]
#                     res_list.append([question, response, model_id, res_["human_score"]])
# save_path = 'sorry-bench-202406-question-answer-pair.json'
# with open(save_path, 'w', encoding='utf-8') as f:
#     json.dump(res_list, f, ensure_ascii=False, indent=2)

# For testing
from dotenv import load_dotenv
load_dotenv()  # Load the .env file

api_key=os.getenv("TOGETHER_API_KEY")
print(f"API Key: {api_key}")  # Should print your API key

from together import Together

client = Together()

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
    max_tokens=2048,
    temperature=0.3,
    top_p=1.0,
    repetition_penalty=1.1,
    stop=["<|eot_id|>","<|eom_id|>"],
    stream=True
)
for token in response:
    if hasattr(token, 'choices'):
        print(token.choices[0].delta.content, end='', flush=True)






