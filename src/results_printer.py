import torch
import os
import sys
PROJECT_PATH = '/home/mhendriksen2/projects/evaluating-cmr-in-vl' if 'mhendriksen2' in os.getcwd() else '/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-vl'
sys.path.append(PROJECT_PATH)
from src.utils.utils import get_config, get_abs_file_paths, get_results_dir, get_project_path
import torch
import os
import argparse
torch.set_num_threads(4)
import pickle
from src.constants.constants import DATASETS, TASKS, MODELS, PERTURBATIONS


def get_mean(dataf, col: str, round_factor=4):
    ans = round(dataf[col].describe().loc["mean"], round_factor)
    if 'recall' in col:
        ans = 100*ans
    return ans

def parse_file_path(path):
    # /Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-mm/results/f30k/clip/t2i/none-results.pkl
    return path.split('/')[-4], path.split('/')[-3], path.split('/')[-2]

def check_string_for_conditions(conditions, s) -> bool:
    ans = []
    for el in conditions:
        if el in s:
            ans.append(True)
        else:
            ans.append(False)
    return all(ans)

def main(args):
    results_dir = get_results_dir()
    results_files = []
    answers = []
    missing_jobs = []
    for dataset in DATASETS:
        for task in TASKS:
            for model in MODELS:
                for perturbation in PERTURBATIONS:
                    filename = f'{perturbation}-results.pkl'
                    filepath = os.path.join(results_dir, dataset, model, task, filename)
                    # results_files.append(filepath)
                    if check_string_for_conditions(conditions=args.c, s=filepath):
                        print(f"File: {filepath}\nTask: {task}\nModel: {model}\nDataset: {dataset}\nR@1, R@5, R@10, DCG:")
                        if os.path.exists(filepath):
                            with open(filepath, 'rb') as f:
                                data = pickle.load(f)
                            ans = f"{get_mean(data, f'{task}_recalls_at_1')}\t{get_mean(data, f'{task}_recalls_at_5')}\t{get_mean(data, f'{task}_recalls_at_10')}\t{get_mean(data, f'{task}_dcgs')}"
                        else:
                            ans = 'None'
                            if 't2i' in task:
                                missing_jobs.append((dataset, task, model, perturbation))
                        print(f'{ans}\n')
                        answers.append(ans)
    print('Printing answers line by line:')
    for ans in answers:
        print(ans)
    
    print('Printing missing jobs line by line:')
    for ans in missing_jobs:
        print(ans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="f30k",
        choices=["coco", "f30k"],
        help="dataset: coco, f30k",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clip",
        choices=["clip", "align", "altclip", "bridgetower", "groupvit"],
        help="model name: clip, blip, flava, beit",
    )
    parser.add_argument(
        "--c",
        '--conditions',
        nargs='+',
        default=[],
        help="list of strings to filter such as model name, dataset type etc.",
    )
    args = parser.parse_args()
    main(args)
