import torch
import os

if torch.cuda.is_available():
    PROJECT_PATH = "/notebooks/evaluating-cmr-in-mm/"
else:
    PROJECT_PATH = "/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-mm/"
import sys
sys.path.append(PROJECT_PATH)

import torch
from munch import Munch
import pandas as pd
import argparse
from tqdm import tqdm
import datetime

torch.set_num_threads(4)
from src.data.dataset import Dataset
from src.utils.dataset_preprocessing import load_json_annotations
from src.retrieval.retriever import Retriever
from src.metrics.recall_at_k import recall_at_k
from src.models.relevance_estimators.clip_based import RelevanceEstimator
from src.metrics.dcg import DCG
from src.utils.dataset_preprocessing import save_results_dataframe
from src.utils.utils import get_config, get_logger, get_model
from src.perturbations.perturbation import Perturbation


class Evaluator(object):

    def __init__(self, config):

        self.config = config
        
        json_file = load_json_annotations(config=self.config)
        self.ds_split = Dataset(config=self.config, split='test', json_file=json_file)
        self.model = get_model(config=self.config)

        if self.config.args.perturbation != 'none':
            self.perturbation = Perturbation(config=self.config)

        self.rel_estimator = RelevanceEstimator(config=self.config, dataset=self.ds_split)
        self.retriever = Retriever(config=self.config, model=self.model)
        self.dcg = DCG(config=self.config, rel_estimator=self.rel_estimator)

        self.logging = self.config.logging


    def i2t(self):

        # load precomputed caption embeddings
        capt_ids, capts, capt_embs = self.model.compute_caption_embeddings(
            ds_split=self.ds_split,
            compute_from_scratch=self.config.args.compute_from_scratch
        )

        i2t_queries = []
        i2t_targets = []
        i2t_retrieved_documents = []
        i2t_scores = []
        i2t_recalls_at_1 = []
        i2t_recalls_at_5 = []
        i2t_recalls_at_10 = []
        i2t_dcgs = []

        print("Image-to-text evaluation...")
        self.logging.info("Image-to-text evaluation...")
        seen_image_ids = []
        for datapoint in self.ds_split:
            # get image query and target
            query = datapoint[1]
            target_filename = datapoint[3]

            if target_filename in seen_image_ids:
                continue
            seen_image_ids.append(target_filename)

            retrieved_caption_ids, scores = self.retriever.retrieve_top_k(
                query=query, documents=capt_embs, documents_names=capt_ids, k=10
            )

            associated_img_ids = [
                self.ds_split.captions[capt_id]["imgid"]
                for capt_id in retrieved_caption_ids
            ]

            # metrics:
            # compute recall at k
            # i2t recall: there is only one correct item in the collection, i.e., total_in_collection=1
            i2t_recall_at_1 = recall_at_k(
                target_filename=target_filename,
                retrieved_documents=associated_img_ids,
                k=1,
                total_in_collection=5,
            )
            i2t_recall_at_5 = recall_at_k(
                target_filename=target_filename,
                retrieved_documents=associated_img_ids,
                k=5,
                total_in_collection=5,
            )
            i2t_recall_at_10 = recall_at_k(
                target_filename=target_filename,
                retrieved_documents=associated_img_ids,
                k=10,
                total_in_collection=5,
            )
            # print('i2t: recalls at 1, 5, 10: ', i2t_recall_at_1, i2t_recall_at_5, i2t_recall_at_10)

            i2t_dcg = self.dcg.compute_dcg(
                query=query,
                target_filename=target_filename,
                retrieved_documents=associated_img_ids,
                caption_ids=retrieved_caption_ids,
            )
            # print('i2t_dcg: ', i2t_dcg)

            i2t_queries.append(query)
            i2t_targets.append(target_filename)

            i2t_retrieved_documents.append(retrieved_caption_ids)
            i2t_scores.append(scores)
            i2t_recalls_at_1.append(i2t_recall_at_1)
            i2t_recalls_at_5.append(i2t_recall_at_5)
            i2t_recalls_at_10.append(i2t_recall_at_10)
            i2t_dcgs.append(i2t_dcg)

            if datapoint[-1] > 0 and datapoint[-1] % 10 == 0:
                ct = datetime.datetime.now()
                print('Time: ', ct)
                print(f"{ct} Progress: {datapoint[-1]}/{len(self.ds_split)}")
                self.logging.info(f"Progress: {datapoint[-1]}/{len(self.ds_split)}")

        data = {
            "i2t_queries": i2t_targets,
            "i2t_targets": i2t_targets,
            "i2t_retrieved_documents": i2t_retrieved_documents,
            "i2t_scores": i2t_scores,
            "i2t_recalls_at_1": i2t_recalls_at_1,
            "i2t_recalls_at_5": i2t_recalls_at_5,
            "i2t_recalls_at_10": i2t_recalls_at_10,
            "i2t_dcgs": i2t_dcgs,
        }

        i2t_results = pd.DataFrame(data=data)

        return i2t_results

    def t2i(self):

        # load precomputed image embeddings
        img_filenames, img_emb = self.model.compute_image_embeddings(
            compute_from_scratch=self.config.args.compute_from_scratch
            )

        t2i_queries = []
        t2i_targets = []
        t2i_retrieved_documents = []
        t2i_scores = []
        t2i_recalls_at_1 = []
        t2i_recalls_at_5 = []
        t2i_recalls_at_10 = []
        t2i_dcgs = []

        print("Text-to-image evaluation...")
        self.logging.info("Text-to-image evaluation...")
        for datapoint in self.ds_split:
            # get textual query and target
            query = datapoint[0]
            if self.config.args.perturbation != 'none':
                # print('Initial caption: ', query)
                try:
                    query = self.perturbation.apply_perturbation_to_caption(query)
                except:
                    print('Problem with the query: ', query)
                # print('new caption: ', query)
                # break
             
            target_filename = datapoint[4]

            retrieved_documents, scores = self.retriever.retrieve_top_k(
                query=query, documents=img_emb, documents_names=img_filenames, k=10
            )

            # metrics:
            # compute recall at k
            # t2i recall: there is only one correct item in the collection, i.e., total_in_collection=1
            t2i_recall_at_1 = recall_at_k(
                target_filename=target_filename,
                retrieved_documents=retrieved_documents,
                k=1,
                total_in_collection=1,
            )
            t2i_recall_at_5 = recall_at_k(
                target_filename=target_filename,
                retrieved_documents=retrieved_documents,
                k=5,
                total_in_collection=1,
            )
            t2i_recall_at_10 = recall_at_k(
                target_filename=target_filename,
                retrieved_documents=retrieved_documents,
                k=10,
                total_in_collection=1,
            )

            t2i_dcg = self.dcg.compute_dcg(
                query=query,
                target_filename=target_filename,
                retrieved_documents=retrieved_documents,
            )
            # print('T2i_dcg: ', t2i_dcg)

            t2i_queries.append(query)
            t2i_targets.append(target_filename)
            t2i_retrieved_documents.append(retrieved_documents)
            t2i_scores.append(scores)
            t2i_recalls_at_1.append(t2i_recall_at_1)
            t2i_recalls_at_5.append(t2i_recall_at_5)
            t2i_recalls_at_10.append(t2i_recall_at_10)
            t2i_dcgs.append(t2i_dcg)

            if datapoint[-1] > 0 and datapoint[-1] % 100 == 0:
                ct = datetime.datetime.now()
                print('Time: ', ct)
                print(f"{ct} Progress: {datapoint[-1]}/{len(self.ds_split)}")
                self.logging.info(f"Progress: {datapoint[-1]}/{len(self.ds_split)}")

        data = {
            "t2i_queries": t2i_queries,
            "t2i_targets": t2i_targets,
            "t2i_retrieved_documents": t2i_retrieved_documents,
            "t2i_scores": t2i_scores,
            "t2i_recalls_at_1": t2i_recalls_at_1,
            "t2i_recalls_at_5": t2i_recalls_at_5,
            "t2i_recalls_at_10": t2i_recalls_at_10,
            "t2i_dcgs": t2i_dcgs,
        }

        t2i_results = pd.DataFrame(data=data)

        return t2i_results
