import glob
import os
import json
import pickle
from typing import List, Generator
from typing import List, Tuple
import numpy as np
from PIL import Image


def get_img_filenames_full_path(img_root: str) -> List[str]:
    return list(glob.glob(str(img_root + '/*.jpg')))


def get_img_filenames(img_root: str) -> List[str]:
    full_paths = get_img_filenames_full_path(img_root=img_root)
    return [os.path.basename(path) for path in full_paths]


def divide_chunks(l: List, n: int) -> Generator:
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_json_annotations(config, augmented=False):
    """
    Load json annotations
    :param config: Config class
    :return:
    """
    if augmented:
        annotation_file = config.dataset.augmentations
    else:
        annotation_file = config.dataset.annotation_file

    file_path = os.path.join(
        config.dataset.root,
        annotation_file
    )
    print('file_path: ', file_path)
    json_file = json.load(open(file_path, 'rb'))
    print('Loaded annotations from ', file_path)

    return json_file


def get_emb_file_path(config, dtype):
    if dtype == 'img':
        filename = config.dataset.img_emb_filename
    elif dtype == 'capt':
        filename = config.dataset.capt_emb_filename
    else:
        raise NotImplementedError

    return os.path.join(config.dataset.root, filename)


def dump_filenames_embs_to_pkl(emb_file_path, data) -> None:
    with open(emb_file_path, 'wb+') as f:
        pickle.dump(data, f)
    print('Saved files to ', emb_file_path)


def load_filenames_embs_from_pkl(emb_file_path):
    with open(emb_file_path, 'rb') as f:
        data = pickle.load(f)
        print('Loaded precomputed filenames and embeddings from ', emb_file_path)
    return data


def get_precomputed_embeddings_path(config, dtype):
    if dtype == 'img':
        filename = config.dataset.img_emb_filename
    elif dtype == 'capt':
        filename = config.dataset.capt_emb_filename
    else:
        raise NotImplementedError

    emb_root = os.path.join(
        config.dataset.root,
        config.dataset.emb_folder
        # , config.args.perturbation
        )
    os.makedirs(emb_root, exist_ok=True)

    path = os.path.join(emb_root, filename)

    print('Loaded embeddings from:', path)
    config.logging.info('Loaded embeddings from:', path)

    return path


def save_results_dataframe(config, dataf, root, filename):
    path = os.path.join(config.results.dir, root)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, str(filename + '.pkl') )
    with open(filepath, 'wb+') as f:
        pickle.dump(dataf, f)
    print('Saved dataframe to ', filepath)
    config.logging.info('Saved dataframe to ', filepath)


def load_results_dataframe(config, filename):
    filepath = os.path.join(config.results.dir, filename + '.pkl')
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print('Loaded results from ', filepath)
    config.logging.info('Loaded results from ', filepath)
    return data
