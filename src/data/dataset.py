import torch
import pickle
import os
from typing import List, Type
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from src.utils.dataset_preprocessing import load_json_annotations


class Dataset(Dataset):

	def __init__(self, config, split, json_file):
		"""
		:param config:
		:param split:
		:param json_file:
		"""
		super().__init__()

		assert split in set(['train', 'val', 'test'])

		self.config = config
		self.split = split

		self.json_file = json_file

		self.captions = {}
		self.images = {}
		self.caption_ids = None

		self._load_annotations_from_json()

		self.augmentations = load_json_annotations(config=self.config, augmented=True)

		if 'aug' in self.config.dataset.name:
			self.augment_captions()

		# TODO: update ds so that perturbations are random

	def __len__(self):
		return len(self.caption_ids)

	def __repr__(self):
		return f'Dataset: {self.json_file["dataset"]}; split: {self.split}'

	def _load_annotations_from_json(self):
		"""
		:return:
		"""
		for image in self.json_file['images']:
			if image['split'] == self.split or (image['split'] == 'restval' and self.split == 'train'):

				self.images[image['imgid']] = {
					'filename': image['filename'],
					'sentids': image['sentids'],
				}

				for sentence in image['sentences']:
					self.captions[sentence['sentid']] = sentence

		self.caption_ids = list(self.captions.keys())

	def __getitem__(self, idx):
		"""
		:param idx:
		:return:
		"""

		caption_id = self.caption_ids[idx]
		caption = self.captions[caption_id]
		raw_caption = caption['raw']

		image_id = caption['imgid']
		image_filename = self.images[image_id]['filename']
		image = Image.open(os.path.join(
			self.config.dataset.root,
			self.config.dataset.img_folder,
			image_filename)).convert('RGB')

		return raw_caption, image, caption_id, image_id, image_filename, idx
	
	def update_caption(self, caption_idx, new_caption):
		self.captions[caption_idx]['raw'] = new_caption

	def 

	def random_perturbations(self):
		"""
		Apply perturbations randomly
		"""

		pass

	def augment_captions(self):
		"""
		caption augmentation
		"""

		dataset = self.json_file
		augmented_dataset = self.augmentations

		for key in tqdm(self.images.keys()):
			# print(f30k_dataset.images[key]['sentids'])
			filename = self.images[key]['filename'].split('.')[0]
			sentids = self.images[key]['sentids']
			# print('filename: ', filename)
			augmented_captions = augmented_dataset[filename]
			# print('augmented_captions: ', augmented_captions)
			for aug_caption, sentid in zip(augmented_captions, sentids):
				self.update_caption(caption_idx=sentid, new_caption=aug_caption)
		print('Finished augmenting the dataset')
	
def get_caption_idx(ds_split: Type[Dataset], capt_ids: List[int]):
    return [ds_split.captions[idx]['raw'] for idx in capt_ids]