import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Type
from sentence_transformers import SentenceTransformer
import munch
from src.data.dataset import Dataset
import os
from typing import List, Type
import PIL
from PIL import Image
from src.utils.dataset_preprocessing import get_img_filenames_full_path, get_img_filenames, divide_chunks, get_precomputed_embeddings_path, load_filenames_embs_from_pkl, get_emb_file_path, dump_filenames_embs_to_pkl
import requests
import torch
import datetime
from PIL import Image
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning

AVAILABLE_MODELS = (
    'BridgeTower/bridgetower-large-itm-mlm-itc'
)

class BridgeTower(nn.Module):


    def __init__(self, config: munch.Munch) -> None:
        """Init function

        Args:
            self (src.models.encoders.bridgetower.BridgeTower): instance of the class
            config (munch.Munch): configuration file (depends on dataset and model)
        """

        super(BridgeTower, self).__init__()

        self.config = config

        self.model_name = self.config.model.name
        assert self.config.model.name in AVAILABLE_MODELS

        self.processor = BridgeTowerProcessor.from_pretrained(self.model_name)
        self.backbone = BridgeTowerForContrastiveLearning.from_pretrained(self.model_name)

    
    def encode(self, x: Union[List[str], List[PIL.Image.Image]]) -> np.ndarray:

        batch = []
        if isinstance(x[0], PIL.Image.Image):
            for image in x:
                inputs = self.processor(
                    images=image,
                    text='dummy_text',
                    return_tensors="pt",
                    padding=True
                )
                outputs = self.backbone(**inputs)
                out_emb = outputs['image_embeds']
                out_emb = out_emb.detach().numpy()
                batch.append(out_emb)

        elif isinstance(x[0], str):
            dummy_img = Image.new('RGB', (640, 480)).convert('RGB')
            for txt in x:
                inputs = self.processor(
                    images=dummy_img,
                    text=txt,
                    return_tensors="pt",
                    padding=True
                    )
                outputs = self.backbone(**inputs)
                out_emb = outputs['text_embeds']
                out_emb = out_emb.detach().numpy()
                batch.append(out_emb)
        else:
            print('Unk input type, options: PIL.Image.Image, str')
        
        batch = np.concatenate(batch, axis=0)

        return batch
    

    def compute_caption_embeddings(self, ds_split: Type[Dataset], chunk_size=1000, compute_from_scratch=False) -> (List[int], List[str], torch.Tensor):
        """Compute caption embeddings for a given dataset

        Args:
            self (src.models.encoders.clip.CLIP): instance of the class
            ds_split (Type[Dataset]): dataset split for extracting caption ids and raw captions

        Returns:
            List[int]: list of caption ids
            List[int]: list of raw captions
            torch.Tensor: tensor of shape (n, m) where n - number of captions, m - caption embedding size
        """
        # check if the path already exists
        emb_path = get_precomputed_embeddings_path(config=self.config, dtype='capt')

        if os.path.exists(emb_path) and not compute_from_scratch:
            print('Caption embeddings are already precomputed')
            caption_ids, capts, capt_emb = load_filenames_embs_from_pkl(emb_file_path=emb_path)
            assert len(caption_ids) == len(capts) == capt_emb.shape[0]
            return caption_ids, capts, capt_emb
        
        else:
            print('Computing caption embeddings...')
            caption_ids = ds_split.caption_ids
            capts = [ds_split.captions[caption_id]['raw'][:self.config.model.max_seq_length] for caption_id in caption_ids]

            # split captions into k chunks, each of size 1000
            capts_chunks = list(divide_chunks(capts, chunk_size))
            # encode chunk by chunk & merge
            capt_embs = []
            for idx, chunk in enumerate(capts_chunks):
                ct = datetime.datetime.now()
                print(f'{ct} Computing embeddings progress: {idx}/{len(capts_chunks)}')
                self.config.logging.info(
                    f'Computing embeddings progress: {idx}/{len(capts_chunks)}'
                    )
                capt_chunk_emb = self.encode(chunk)
                capt_embs.append(capt_chunk_emb)
            capt_emb = np.concatenate(capt_embs, axis=0)
            capt_emb = torch.from_numpy(capt_emb)

            assert len(capts) == capt_emb.shape[0]

            data = (caption_ids, capts, capt_emb)

            print('Saving captions...')
            dump_filenames_embs_to_pkl(emb_path, data)

            return data
        
        
    def compute_image_embeddings(self, chunk_size=1000, compute_from_scratch=False) -> (List[str], torch.Tensor):
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
        if os.path.exists(emb_path) and not compute_from_scratch:
            print('Image embeddigns are already precomputed')
            data = load_filenames_embs_from_pkl(emb_file_path=emb_path)
            return data
        else:
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
            # print('imgs_full_paths_chunks example ', imgs_full_paths_chunks[0][4])
            for idx, chunk in enumerate(imgs_full_paths_chunks):
                ct = datetime.datetime.now()
                print(f'{ct} Computing embeddings progress: {idx}/{len(imgs_full_paths_chunks)}')
                self.config.logging.info(
                    f'Computing embeddings progress: {idx}/{len(imgs_full_paths_chunks)}'
                    )
                images = [Image.open(filepath).convert('RGB') for filepath in chunk]
                # print('Image example : ', images[0])
                img_emb = self.encode(images)
                img_embs.append(img_emb)
            img_emb = np.concatenate(img_embs, axis=0)
            img_emb = torch.from_numpy(img_emb)

            # assert len(img_filenames) == img_emb.shape[0]

            data = (img_filenames, img_emb)

            print('Saving images...')
            dump_filenames_embs_to_pkl(emb_path, data)

            return data
