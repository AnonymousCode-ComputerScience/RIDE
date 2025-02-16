import argparse
import json
import os
from collections import defaultdict

PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# print(PROJECT_ABSOLUTE_PATH)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath',
                        default="just-eval/leaderboard/eval_results/gpt-4o/alpaca_eval_for_scoring/vllm-value_impact_rewrite_upgrade_icl_16_8_sorry7-Llama-2-7b-hf_for-score.score_multi.json",
                        type=str)
    parser.add_argument('--safety_filepath',
                        default="just-eval/leaderboard/eval_results/gpt-4o/vllm-value_impact_rewrite_upgrade_icl_16_8_sorry7-variant-Llama-2-7b-hf.score_safety.json",
                        type=str)
    return parser.parse_args()


def count_good_divided(data, filepath):
    res_helpful = {"helpfulness": [], "clarity": [], "factuality": [], "depth": [], "engagement": [],
                   "helpfulness_count": {}, "clarity_count": {}, "factuality_count": {},
                   "depth_count": {}, "engagement_count": {}}
    for index_, item_ in enumerate(data):
        if 'parsed_result' in item_ and item_["parsed_result"] is not None:
            if "helpfulness" in item_["parsed_result"] and "score" in item_["parsed_result"]["helpfulness"]:
                if item_["parsed_result"]["helpfulness"]["score"] == "5" or item_["parsed_result"]["helpfulness"][
                    "score"] == 5:
                    res_helpful["helpfulness"].append(item_["output_cand"])
            if "clarity" in item_["parsed_result"] and "score" in item_["parsed_result"]["clarity"]:
                if item_["parsed_result"]["clarity"]["score"] == "5" or item_["parsed_result"]["clarity"]["score"] == 5:
                    res_helpful["clarity"].append(item_["output_cand"])
            if "factuality" in item_["parsed_result"] and "score" in item_["parsed_result"]["factuality"]:
                if item_["parsed_result"]["factuality"]["score"] == "5" or item_["parsed_result"]["factuality"][
                    "score"] == 5:
                    res_helpful["factuality"].append(item_["output_cand"])
            if "depth" in item_["parsed_result"] and "score" in item_["parsed_result"]["depth"]:
                if item_["parsed_result"]["depth"]["score"] == "5" or item_["parsed_result"]["depth"]["score"] == 5:
                    res_helpful["depth"].append(item_["output_cand"])
            if "engagement" in item_["parsed_result"] and "score" in item_["parsed_result"]["engagement"]:
                if item_["parsed_result"]["engagement"]["score"] == "5" or item_["parsed_result"]["engagement"][
                    "score"] == 5:
                    res_helpful["engagement"].append(item_["output_cand"])

    res_helpful["helpfulness_count"]["unigram"] = defaultdict(int)
    res_helpful["helpfulness_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["helpfulness"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["helpfulness_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["helpfulness_count"]["bigram"][bigram] += 1

    res_helpful["clarity_count"]["unigram"] = defaultdict(int)
    res_helpful["clarity_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["clarity"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["clarity_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["clarity_count"]["bigram"][bigram] += 1

    res_helpful["factuality_count"]["unigram"] = defaultdict(int)
    res_helpful["factuality_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["factuality"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["factuality_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["factuality_count"]["bigram"][bigram] += 1

    res_helpful["depth_count"]["unigram"] = defaultdict(int)
    res_helpful["depth_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["depth"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["depth_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["depth_count"]["bigram"][bigram] += 1

    res_helpful["engagement_count"]["unigram"] = defaultdict(int)
    res_helpful["engagement_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["engagement"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["engagement_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["engagement_count"]["bigram"][bigram] += 1

    res_helpful["helpfulness_count"]["unigram"] = dict(
        sorted(res_helpful["helpfulness_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["helpfulness_count"]["bigram"] = dict(
        sorted(res_helpful["helpfulness_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    res_helpful["clarity_count"]["unigram"] = dict(
        sorted(res_helpful["clarity_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["clarity_count"]["bigram"] = dict(
        sorted(res_helpful["clarity_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    res_helpful["factuality_count"]["unigram"] = dict(
        sorted(res_helpful["factuality_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["factuality_count"]["bigram"] = dict(
        sorted(res_helpful["factuality_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    res_helpful["depth_count"]["unigram"] = dict(
        sorted(res_helpful["depth_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["depth_count"]["bigram"] = dict(
        sorted(res_helpful["depth_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    res_helpful["engagement_count"]["unigram"] = dict(
        sorted(res_helpful["engagement_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["engagement_count"]["bigram"] = dict(
        sorted(res_helpful["engagement_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    with open(filepath.replace('.json', '_count_good.json'), 'w') as json_file:
        json.dump(res_helpful, json_file, indent=4)


def count_average(data, filepath, window=10):
    res_helpful = {
        "helpful": [], "helpful_count": {}, "helpful_count_freq": {}
    }
    res_helpless = {
        "helpless": [], "helpless_count": {}, "helpless_count_freq": {}
    }
    for index_, item_ in enumerate(data):
        if 'parsed_result' in item_ and item_["parsed_result"] is not None:
            scores = 0
            if "helpfulness" in item_["parsed_result"] and "score" in item_["parsed_result"]["helpfulness"]:
                score = int(item_["parsed_result"]["helpfulness"]["score"])
                scores += score
            else:
                scores += 0

            if "clarity" in item_["parsed_result"] and "score" in item_["parsed_result"]["clarity"]:
                score = int(item_["parsed_result"]["clarity"]["score"])
                scores += score
            else:
                scores += 0

            if "factuality" in item_["parsed_result"] and "score" in item_["parsed_result"]["factuality"]:
                score = int(item_["parsed_result"]["factuality"]["score"])
                scores += score
            else:
                scores += 0

            if "depth" in item_["parsed_result"] and "score" in item_["parsed_result"]["depth"]:
                score = int(item_["parsed_result"]["depth"]["score"])
                scores += score
            else:
                scores += 0

            if "engagement" in item_["parsed_result"] and "score" in item_["parsed_result"]["engagement"]:
                score = int(item_["parsed_result"]["engagement"]["score"])
                scores += score
            else:
                scores += 0

            if scores / 5.0 >= 4.4:
                res_helpful["helpful"].append(item_["output_cand"])

            elif scores / 5.0 <= 2.6:
                res_helpless["helpless"].append(item_["output_cand"])

    res_helpful["helpful_count"]["unigram"] = defaultdict(int)
    res_helpful["helpful_count"]["bigram"] = defaultdict(int)

    res_helpful["helpful_count_freq"]["unigram"] = {}
    res_helpful["helpful_count_freq"]["bigram"] = {}

    for sentence in res_helpful["helpful"]:
        tokens = sentence.strip().split()[:window]
        for token in tokens:
            res_helpful["helpful_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["helpful_count"]["bigram"][bigram] += 1

    res_helpless["helpless_count"]["unigram"] = defaultdict(int)
    res_helpless["helpless_count"]["bigram"] = defaultdict(int)

    res_helpless["helpless_count_freq"]["unigram"] = {}
    res_helpless["helpless_count_freq"]["bigram"] = {}

    for sentence in res_helpless["helpless"]:
        tokens = sentence.strip().split()[:window]
        for token in tokens:
            res_helpless["helpless_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpless["helpless_count"]["bigram"][bigram] += 1

    res_helpful["helpful_count"]["unigram"] = dict(
        sorted(res_helpful["helpful_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["helpful_count"]["bigram"] = dict(
        sorted(res_helpful["helpful_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:300])

    len_ = len(res_helpful["helpful"])
    for key_, value_ in res_helpful["helpful_count"]["unigram"].items():
        res_helpful["helpful_count_freq"]["unigram"][key_] = float(value_ / len_)
    res_helpful["helpful_count_freq"]["unigram"] = dict(
        sorted(res_helpful["helpful_count_freq"]["unigram"].items(), key=lambda item: item[1], reverse=True))

    for key_, value_ in res_helpful["helpful_count"]["bigram"].items():
        res_helpful["helpful_count_freq"]["bigram"][key_] = float(value_ / len_)
    res_helpful["helpful_count_freq"]["bigram"] = dict(
        sorted(res_helpful["helpful_count_freq"]["bigram"].items(), key=lambda item: item[1], reverse=True))

    res_helpless["helpless_count"]["unigram"] = dict(
        sorted(res_helpless["helpless_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpless["helpless_count"]["bigram"] = dict(
        sorted(res_helpless["helpless_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:300])

    len_ = len(res_helpless["helpless"])
    for key_, value_ in res_helpless["helpless_count"]["unigram"].items():
        res_helpless["helpless_count_freq"]["unigram"][key_] = float(value_ / len_)
    res_helpless["helpless_count_freq"]["unigram"] = dict(
        sorted(res_helpless["helpless_count_freq"]["unigram"].items(), key=lambda item: item[1], reverse=True))

    for key_, value_ in res_helpless["helpless_count"]["bigram"].items():
        res_helpless["helpless_count_freq"]["bigram"][key_] = float(value_ / len_)
    res_helpless["helpless_count_freq"]["bigram"] = dict(
        sorted(res_helpless["helpless_count_freq"]["bigram"].items(), key=lambda item: item[1], reverse=True))

    with open(filepath.replace('.json', '_count_good_average.json'), 'w') as json_file:
        json.dump(res_helpful, json_file, indent=4)

    with open(filepath.replace('.json', '_count_bad_average.json'), 'w') as json_file:
        json.dump(res_helpless, json_file, indent=4)


def count_average_safety(data, filepath, window=10):
    res_helpful = {
        "helpful": [], "helpful_count": {}, "helpful_count_freq": {}
    }
    res_helpless = {
        "helpless": [], "helpless_count": {}, "helpless_count_freq": {}
    }
    for index_, item_ in enumerate(data):
        if 'parsed_result' in item_ and item_["parsed_result"] is not None:
            if "safety" in item_["parsed_result"] and "score" in item_["parsed_result"]["safety"]:
                score = int(item_["parsed_result"]["safety"]["score"])
                if score == 5:
                    res_helpful["helpful"].append(item_["output_cand"])
                if score <= 3:
                    res_helpless["helpless"].append(item_["output_cand"])

    res_helpful["helpful_count"]["unigram"] = defaultdict(int)
    res_helpful["helpful_count"]["bigram"] = defaultdict(int)

    res_helpful["helpful_count_freq"]["unigram"] = {}
    res_helpful["helpful_count_freq"]["bigram"] = {}

    for sentence in res_helpful["helpful"]:
        tokens = sentence.strip().split()[:window]
        for token in tokens:
            res_helpful["helpful_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["helpful_count"]["bigram"][bigram] += 1

    res_helpless["helpless_count"]["unigram"] = defaultdict(int)
    res_helpless["helpless_count"]["bigram"] = defaultdict(int)

    res_helpless["helpless_count_freq"]["unigram"] = {}
    res_helpless["helpless_count_freq"]["bigram"] = {}

    for sentence in res_helpless["helpless"]:
        tokens = sentence.strip().split()[:window]
        for token in tokens:
            res_helpless["helpless_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpless["helpless_count"]["bigram"][bigram] += 1

    res_helpful["helpful_count"]["unigram"] = dict(
        sorted(res_helpful["helpful_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["helpful_count"]["bigram"] = dict(
        sorted(res_helpful["helpful_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:300])

    len_ = len(res_helpful["helpful"])
    for key_, value_ in res_helpful["helpful_count"]["unigram"].items():
        res_helpful["helpful_count_freq"]["unigram"][key_] = float(value_ / len_)
    res_helpful["helpful_count_freq"]["unigram"] = dict(
        sorted(res_helpful["helpful_count_freq"]["unigram"].items(), key=lambda item: item[1], reverse=True))

    for key_, value_ in res_helpful["helpful_count"]["bigram"].items():
        res_helpful["helpful_count_freq"]["bigram"][key_] = float(value_ / len_)
    res_helpful["helpful_count_freq"]["bigram"] = dict(
        sorted(res_helpful["helpful_count_freq"]["bigram"].items(), key=lambda item: item[1], reverse=True))

    res_helpless["helpless_count"]["unigram"] = dict(
        sorted(res_helpless["helpless_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpless["helpless_count"]["bigram"] = dict(
        sorted(res_helpless["helpless_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:300])

    len_ = len(res_helpless["helpless"])
    for key_, value_ in res_helpless["helpless_count"]["unigram"].items():
        res_helpless["helpless_count_freq"]["unigram"][key_] = float(value_ / len_)
    res_helpless["helpless_count_freq"]["unigram"] = dict(
        sorted(res_helpless["helpless_count_freq"]["unigram"].items(), key=lambda item: item[1], reverse=True))

    for key_, value_ in res_helpless["helpless_count"]["bigram"].items():
        res_helpless["helpless_count_freq"]["bigram"][key_] = float(value_ / len_)
    res_helpless["helpless_count_freq"]["bigram"] = dict(
        sorted(res_helpless["helpless_count_freq"]["bigram"].items(), key=lambda item: item[1], reverse=True))

    with open(filepath.replace('.json', '_count_safe_average.json'), 'w') as json_file:
        json.dump(res_helpful, json_file, indent=4)

    with open(filepath.replace('.json', '_count_unsafe_average.json'), 'w') as json_file:
        json.dump(res_helpless, json_file, indent=4)

def count_bad_divided(data, filepath):
    res_helpful = {"helpfulness": [], "clarity": [], "factuality": [], "depth": [], "engagement": [],
                   "helpfulness_count": {}, "clarity_count": {}, "factuality_count": {},
                   "depth_count": {}, "engagement_count": {}}
    for index_, item_ in enumerate(data):
        if 'parsed_result' in item_ and item_["parsed_result"] is not None:
            if "helpfulness" in item_["parsed_result"] and "score" in item_["parsed_result"]["helpfulness"]:
                if item_["parsed_result"]["helpfulness"]["score"] == "1" or item_["parsed_result"]["helpfulness"][
                    "score"] == 1 or item_["parsed_result"]["helpfulness"]["score"] == "2" or item_["parsed_result"]["helpfulness"][
                    "score"] == 2:
                    res_helpful["helpfulness"].append(item_["output_cand"])
            if "clarity" in item_["parsed_result"] and "score" in item_["parsed_result"]["clarity"]:
                if item_["parsed_result"]["clarity"]["score"] == "1" or item_["parsed_result"]["clarity"]["score"] == 1 or item_["parsed_result"]["clarity"]["score"] == "2" or item_["parsed_result"]["clarity"]["score"] == 2:
                    res_helpful["clarity"].append(item_["output_cand"])
            if "factuality" in item_["parsed_result"] and "score" in item_["parsed_result"]["factuality"]:
                if item_["parsed_result"]["factuality"]["score"] == "1" or item_["parsed_result"]["factuality"][
                    "score"] == 1 or item_["parsed_result"]["factuality"]["score"] == "2" or item_["parsed_result"]["factuality"][
                    "score"] == 2:
                    res_helpful["factuality"].append(item_["output_cand"])
            if "depth" in item_["parsed_result"] and "score" in item_["parsed_result"]["depth"]:
                if item_["parsed_result"]["depth"]["score"] == "1" or item_["parsed_result"]["depth"]["score"] == 1 or item_["parsed_result"]["depth"]["score"] == "2" or item_["parsed_result"]["depth"]["score"] == 2:
                    res_helpful["depth"].append(item_["output_cand"])
            if "engagement" in item_["parsed_result"] and "score" in item_["parsed_result"]["engagement"]:
                if item_["parsed_result"]["engagement"]["score"] == "1" or item_["parsed_result"]["engagement"][
                    "score"] == 1 or item_["parsed_result"]["engagement"]["score"] == "2" or item_["parsed_result"]["engagement"][
                    "score"] == 2:
                    res_helpful["engagement"].append(item_["output_cand"])

    res_helpful["helpfulness_count"]["unigram"] = defaultdict(int)
    res_helpful["helpfulness_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["helpfulness"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["helpfulness_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["helpfulness_count"]["bigram"][bigram] += 1

    res_helpful["clarity_count"]["unigram"] = defaultdict(int)
    res_helpful["clarity_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["clarity"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["clarity_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["clarity_count"]["bigram"][bigram] += 1

    res_helpful["factuality_count"]["unigram"] = defaultdict(int)
    res_helpful["factuality_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["factuality"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["factuality_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["factuality_count"]["bigram"][bigram] += 1

    res_helpful["depth_count"]["unigram"] = defaultdict(int)
    res_helpful["depth_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["depth"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["depth_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["depth_count"]["bigram"][bigram] += 1

    res_helpful["engagement_count"]["unigram"] = defaultdict(int)
    res_helpful["engagement_count"]["bigram"] = defaultdict(int)
    for sentence in res_helpful["engagement"]:
        tokens = sentence.strip().split()[:5]
        for token in tokens:
            res_helpful["engagement_count"]["unigram"][token] += 1
        bigrams = [' '.join((tokens[i], tokens[i + 1])) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            res_helpful["engagement_count"]["bigram"][bigram] += 1

    res_helpful["helpfulness_count"]["unigram"] = dict(
        sorted(res_helpful["helpfulness_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["helpfulness_count"]["bigram"] = dict(
        sorted(res_helpful["helpfulness_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    res_helpful["clarity_count"]["unigram"] = dict(
        sorted(res_helpful["clarity_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["clarity_count"]["bigram"] = dict(
        sorted(res_helpful["clarity_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    res_helpful["factuality_count"]["unigram"] = dict(
        sorted(res_helpful["factuality_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["factuality_count"]["bigram"] = dict(
        sorted(res_helpful["factuality_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    res_helpful["depth_count"]["unigram"] = dict(
        sorted(res_helpful["depth_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["depth_count"]["bigram"] = dict(
        sorted(res_helpful["depth_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    res_helpful["engagement_count"]["unigram"] = dict(
        sorted(res_helpful["engagement_count"]["unigram"].items(), key=lambda item: item[1], reverse=True)[:100])
    res_helpful["engagement_count"]["bigram"] = dict(
        sorted(res_helpful["engagement_count"]["bigram"].items(), key=lambda item: item[1], reverse=True)[:100])

    with open(filepath.replace('.json', '_count_bad.json'), 'w') as json_file:
        json.dump(res_helpful, json_file, indent=4)


if __name__ == "__main__":
    args = parse_args()
    # filepath = PROJECT_ABSOLUTE_PATH + '/' + args.filepath
    #
    # with open(filepath, 'r') as file:
    #     data = json.load(file)

    safety_filepath = PROJECT_ABSOLUTE_PATH + '/' + args.safety_filepath
    with open(safety_filepath, 'r') as file:
        safety_data = json.load(file)

    # count_good_divided(data, filepath)
    #
    # count_bad_divided(data, filepath)
    #
    # count_average(data, filepath)

    count_average_safety(safety_data, safety_filepath)
