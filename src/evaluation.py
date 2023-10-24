import os
PROJECT_PATH = '/home/mhendriksen2/projects/evaluating-cmr-in-vl' if 'mhendriksen2' in os.getcwd() else '/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-vl'
import sys
sys.path.append(PROJECT_PATH)
import torch
from munch import Munch
import argparse

torch.set_num_threads(4)
from src.utils.dataset_preprocessing import save_results_dataframe
from src.utils.utils import get_config, get_logger
from src.evaluation.evaluator import Evaluator


def main(args):

    print("Args: ", args)
    config = get_config(dataset=args.dataset, model=args.model)

    config.args = Munch(
        dataset=args.dataset,
        model=args.model, 
        task=args.task,
        perturbation=args.perturbation,
        compute_from_scratch=args.compute_from_scratch
        )
    
    logging = get_logger(config=config)
    config.logging = logging

    evaluator = Evaluator(config=config)

    if args.task == 'i2t':
        results = evaluator.i2t()
    elif args.task == 't2i':
        results = evaluator.t2i()
    else:
        print('Uknown task type, options: i2t, t2i')
    print(results.describe())
    print(results.head())
    config.logging.info(results.describe())
    config.logging.info(results.head())

    config.logging.info("Saving the results...")
    print("Saving the results...")
    save_results_dataframe(
        config=config, dataf=results,
        root=f'{args.dataset}/{args.model}/{args.task}',
        filename=f"{args.perturbation}-results"
    )
    config.logging.info("Done!")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="f30k",
        choices=["coco", "f30k", "f30k_aug", "coco_aug"],
        help="dataset: coco, f30k",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clip",
        choices=["clip", "align", "altclip", "groupvit"],
        help="model name: clip, align, altclip, groupvit",
    )
    parser.add_argument(
        "--task",
        type=str,
        default='t2i',
        choices=["t2i", "i2t"],
        help="Task type: t2i, i2t",
    )
    parser.add_argument(
        "--compute_from_scratch",
        action='store_true',
        help="Compute embeddings from scratch?",
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        default="none",
        choices=[
            "none",
            'char_swap',
            'missing_char',
            'extra_char',
            'nearby_char',
            'probability_based_letter_change',
            "synonym_noun",
            'synonym_adj',
            "distraction_true",
            "distraction_false",
            "shuffle_nouns_and_adj",
            'shuffle_all_words',
            'shuffle_allbut_nouns_and_adj',
            'shuffle_within_trigrams',
            'shuffle_trigrams'
            ],
        help="perturbation type: none, typos, etc.",
    )
    args = parser.parse_args()
    main(args)
