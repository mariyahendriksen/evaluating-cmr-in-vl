import torch
import torch.nn as nn
import numpy as np
from typing import Union
from sentence_transformers import SentenceTransformer
import munch
from src.data.dataset import Dataset
import os
from typing import List, Type
import PIL
from PIL import Image
from src.utils.dataset_preprocessing import get_img_filenames_full_path, get_img_filenames, divide_chunks, get_precomputed_embeddings_path, load_filenames_embs_from_pkl, get_emb_file_path, dump_filenames_embs_to_pkl

AVAILABLE_MODELS = (
    'clip-ViT-B-32',
    'clip-ViT-B-16',
    'clip-ViT-L-14'
)


class CLIP(nn.Module):

    def __init__(self, config: munch.Munch) -> None:
        """Init function

        Args:
            self (src.models.encoders.clip.CLIP): instance of the class
            config (munch.Munch): configuration file (depends on dataset and model)
        """

        super(CLIP, self).__init__()

        self.config = config

        self.model_name = self.config.model.name
        assert self.config.model.name in AVAILABLE_MODELS

        self.backbone = SentenceTransformer(self.model_name)

    def encode(self, x: Union[str, PIL.Image.Image], **kwargs) -> np.ndarray:
        """Encoding function

        Args:
            self (src.models.encoders.clip.CLIP): instance of the class
            x (Union[str, PIL.Image.Image]): input, can be either a caption or an image

        Returns:
            np.ndarray: model output; tensor of shape (n, m) where n - number of inputs, m - embedding size
        """

        return self.backbone.encode(x, **kwargs)

    def compute_caption_embeddings(self, ds_split: Type[Dataset], compute_from_scratch) -> (List[int], List[str], torch.Tensor):
        """Compute caption embeddings for a given dataset

        Args:
            self (src.models.encoders.clip.CLIP): instance of the class
            ds_split (Type[Dataset]): dataset split for extracting caption ids and raw captions

        Returns:
            List[int]: list of caption ids
            List[int]: list of raw captions
            torch.Tensor: tensor of shape (n, m) where n - number of captions, m - caption embedding size
        """

        # generate a path
        emb_path = get_precomputed_embeddings_path(config=self.config, dtype='capt')
        print('Target embedding path: ', emb_path)

        if compute_from_scratch:
            print('Computing caption embeddings...')
            caption_ids = ds_split.caption_ids
            capt_emb = self.encode(
                [ds_split.captions[caption_id]['raw'][:self.config.model.max_seq_length] for caption_id in caption_ids],
                batch_size=128,
                convert_to_tensor=True,
                show_progress_bar=False
                )

            capts = [ds_split.captions[caption_id]['raw'] for caption_id in caption_ids]
            # assert len(caption_ids) == len(capts) == capt_emb.shape[0]

            data = (caption_ids, capts, capt_emb)

            print('Saving captions...')

            dump_filenames_embs_to_pkl(emb_path, data)

            return data
        
        else:
            print('Caption embeddings are already precomputed')
            caption_ids, capts, capt_emb = load_filenames_embs_from_pkl(emb_file_path=emb_path)
            assert len(caption_ids) == len(capts) == capt_emb.shape[0]
            return caption_ids, capts, capt_emb

    def compute_image_embeddings(self, compute_from_scratch, chunk_size=1000) -> (List[str], torch.Tensor):
        """Compute image embeddings for a given dataset
        Args:
            dataset_name (str) : dataset name
            chunk_size (int) : size of the chunk used preprocess the data
        Returns:
            List[str]: list of strings that indicate img_filenames
            torch.Tensor: tensor of embeddings of shape (n, m) where n - number of images, m - embedding size
        """

        # check if the path already exists
        emb_path = get_precomputed_embeddings_path(config=self.config, dtype='img')
        print('Target embedding path: ', emb_path)
        if compute_from_scratch:
            print('Computing image embeddings...')
            # generate full path to images
            img_root = os.path.join(
                self.config.dataset.root,
                self.config.dataset.img_folder)
            images_full_paths = get_img_filenames_full_path(img_root=img_root)
            img_filenames = get_img_filenames(img_root=img_root)

            # split full filepaths into k chunks, each of size 1000
            imgs_full_paths_chunks = list(divide_chunks(images_full_paths, chunk_size))

            # encode chunk by chunk & merge
            img_embs = []
            for idx, chunk in enumerate(imgs_full_paths_chunks):
                print(f'Progress: {idx}/{len(imgs_full_paths_chunks)}')
                img_emb = self.encode(
                    [Image.open(filepath).convert('RGB') for filepath in chunk],
                    batch_size=128,
                    convert_to_tensor=True,
                    show_progress_bar=False
                    )
                img_embs.append(img_emb)
            img_emb = torch.concat(img_embs, dim=0)

            assert len(img_filenames) == img_emb.shape[0]

            data = (img_filenames, img_emb)

            print('Saving images...')
            dump_filenames_embs_to_pkl(emb_path, data)

            return data

        else:
            print('Image embeddigns are already precomputed')
            data = load_filenames_embs_from_pkl(emb_file_path=emb_path)
            return data

