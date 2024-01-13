"""
This script takes a dataset as an input and returns its version with perturbations applied on captions
"""
import argparse
import sys
project_path = '/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-vl'
if project_path not in sys.path:
    sys.path.append(project_path)

from tqdm import tqdm
from src.perturbations.perturbation import Perturbation
from src.utils.dataset_preprocessing import load_json_annotations
from munch import Munch
from src.data.dataset import Dataset
import torch
import os


def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def split_list_into_k_sublists(l, k):
    """
    l: List
    k: number of perturbations
    """
    # initial sublist size
    sublist_size = len(l) // k
    # print('sublist_size: ', sublist_size)

    if len(l) % k > 0:
        sublist_size += 1

    ans = list(divide_chunks(l, sublist_size))

    assert len(ans) == k

    return ans


ALL_PERTURBATIONS  = [
    'char_swap',
    'missing_char',
    'extra_char',
    'nearby_char',
    # 'probability_based_letter_change',
    'synonym_noun',
    'synonym_adj',
    'distraction_true',
    'distraction_false',
    'shuffle_nouns_and_adj',
    'shuffle_all_words',
    'shuffle_allbut_nouns_and_adj',
    'shuffle_within_trigrams',
    'shuffle_trigrams'
]

INCREASING_COMPLEXITY_PERTURBATIONS  = [
    'synonym_adj',
    'distraction_false',
    'distraction_true',
    'synonym_noun',
    'extra_char',
    'shuffle_trigrams',
    'missing_char',
    'shuffle_within_trigrams',
    'shuffle_allbut_nouns_and_adj',
    'nearby_char',
    'char_swap',
    'shuffle_nouns_and_adj',
    'shuffle_all_words'
]


def main(args):
    print('ARGS: ', args)
    dataset = args.dataset

    model = 'align'
    task = 't2i'
    perturbation = 'all_ordered'
    config_path = '/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-vl/config/align/f30k/development_local.yaml'
    with open(config_path, 'rb') as f:
        config = Munch.fromYAML(f)

    # TODO: align config ds with the args dataset to ensure augmentation
    # TODO: test with assert
    json_file = load_json_annotations(config)

    config.args = Munch(
        dataset=dataset,
        model=model, 
        task=task,
        perturbation=perturbation,
        compute_from_scratch=False
    )
    assert dataset == config.args.dataset
    
    ds_split = Dataset(config=config, split='test', json_file=json_file)

    if 'aug' in dataset:
        ds_split.augment_captions()

    img_ids = (list(ds_split.images.keys()))

    k = len(INCREASING_COMPLEXITY_PERTURBATIONS)
    k_img_ids_sublists = split_list_into_k_sublists(l=img_ids, k=k)
    print(f'Split ds into {k} sublists, each of length {len(k_img_ids_sublists[0])}')

    for idx, (perturbation_type, sublist) in enumerate(zip(INCREASING_COMPLEXITY_PERTURBATIONS, k_img_ids_sublists)):
        print(f"{idx}/{len(INCREASING_COMPLEXITY_PERTURBATIONS)}", perturbation_type)
        # initialize perturbation
        perturbation = Perturbation(
            config=config,
            perturbation_type=perturbation_type)
        
        for img_id in tqdm(sublist):
            sentids = ds_split.images[img_id]['sentids']
            for sentid in sentids:
                # print(sentid)
                tmp_capt = ds_split.captions[sentid]['raw']
                # print('Original caption: ', tmp_capt)
                perturbed_caption = perturbation.apply_perturbation_to_caption(tmp_capt)
                ds_split.update_caption(caption_idx=sentid, new_caption=perturbed_caption)
                # print('Modified caption: ', perturbed_caption)

    results_file = os.path.join(args.datasets_path, f"{args.dataset}.pth")
    torch.save(ds_split, results_file)
    print('Saved to ', results_file)


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
        "--datasets_path",
        type=str,
        default="./deliverables/datasets/",
        help="folder to save datasets in",
    )
    args = parser.parse_args()
    main(args)
