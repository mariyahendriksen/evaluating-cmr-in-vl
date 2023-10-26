import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import munch
import os
import numpy as np
from typing import List, Callable
import PIL
from PIL import Image
from src.utils.dataset_preprocessing import get_img_filenames_full_path, get_img_filenames, divide_chunks, get_precomputed_embeddings_path, load_filenames_embs_from_pkl
from src.utils.image_processing import get_image

AVAILABLE_MODELS = (
    'clip-ViT-B-32',
    'clip-ViT-B-16',
    'clip-ViT-L-14'
)

class RelevanceEstimator(nn.Module):

    def __init__(self, config, dataset) -> None:

        super(RelevanceEstimator, self).__init__()

        self.config = config

        self.dataset = dataset

        self.model_name = self.config.dcg.relevance_estimator.name
        assert self.model_name in AVAILABLE_MODELS

        self.backbone = SentenceTransformer(self.model_name)
        self.sim_func = self.get_sim_score()

    def encode(self, x, **kwargs) -> np.ndarray:
        return self.backbone.encode(x[:self.config.model.max_seq_length], **kwargs)

    def get_sim_score(self) -> Callable:
        if self.config.dcg.relevance_estimator.sim_score == 'cosine':
            return util.cos_sim
        else:
            raise NotImplementedError

    def compute_relevance_estimation_t2i(self, query: str, document: str) -> float:

        assert type(query) == type(document)

        img = get_image(config=self.config, filename=document)

        query = query[:self.config.model.max_seq_length]
        query_emb = self.encode(query)
        document_emb = self.encode(img)
        return round(self.sim_func(query_emb, document_emb).item(), 4)

    def get_caption(self, caption_id: int) -> str:
        return self.dataset.captions[caption_id]['raw']

    def compute_relevance_estimation_i2t(self, query: str, document: str, caption_id: int) -> float:
        # print('Query: ', query)
        # print('document: ', document)
        # print('capt_id', caption_id)
        query_emb = self.encode(query)
        caption = self.get_caption(caption_id=caption_id)[:self.config.model.max_seq_length]
        # print('Caption: ', caption)
        caption_emb = self.encode(caption)

        # print('score: ', round(self.sim_func(query_emb, caption_emb).item(), 4))

        return round(self.sim_func(query_emb, caption_emb).item(), 4)
        # raise NotImplementedError
