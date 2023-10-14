import torch
import os
import sys

PROJECT_PATH = '/home/mhendriksen2/projects/evaluating-cmr-in-vl'
sys.path.append(PROJECT_PATH)
from src.utils.utils import get_config, get_abs_file_paths, get_results_dir, get_project_path


import torch
import os
import argparse
torch.set_num_threads(4)
import pickle


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
    # if torch.cuda.is_available():
    #     results_dir = '/notebooks/evaluating-cmr-in-mm/results'
    # else:
    #     results_dir = '/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-mm/results'
    results_dir = get_results_dir()

    print('results_root: ', results_dir)
    results_files = list(get_abs_file_paths(results_dir))
    results_files = [file for file in results_files if file.endswith('.pkl')]
    results_files.sort()
    # print('results_files: ', results_files)

    for filepath in results_files:
        # print('filepath', filepath)
        if check_string_for_conditions(conditions=args.c, s=filepath):
            dataset, model, task = parse_file_path(path=filepath)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(
                f"""
                File: {filepath}
                Df size: {data.shape}
                Task: {task}
                Model: {model}
                Dataset: {dataset}
                R@1, R@5, R@10, DCG:
                {get_mean(data, f'{task}_recalls_at_1')}\t{get_mean(data, f'{task}_recalls_at_5')}\t{get_mean(data, f'{task}_recalls_at_10')}\t{get_mean(data, f'{task}_dcgs')}
                    """
                )


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
