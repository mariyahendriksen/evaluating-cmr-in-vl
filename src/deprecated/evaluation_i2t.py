import torch
import os

if torch.cuda.is_available():
    PROJECT_PATH = "/notebooks/evaluating-cmr-in-mm/"
    os.environ["http_proxy"] = "http://devproxy.bloomberg.com:82"
    os.environ["https_proxy"] = "http://devproxy.bloomberg.com:82"
else:
    PROJECT_PATH = "/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-mm/"
import sys
sys.path.append(PROJECT_PATH)

import torch
from munch import Munch
import pandas as pd
import argparse
from tqdm import tqdm

torch.set_num_threads(4)
from src.data.dataset import Dataset
from src.utils.dataset_preprocessing import load_json_annotations
from src.retrieval.retriever import Retriever
from src.metrics.recall_at_k import recall_at_k
from src.models.relevance_estimators.clip_based import RelevanceEstimator
from src.metrics.dcg import DCG
from src.utils.dataset_preprocessing import save_results_dataframe
from src.utils.utils import get_config, get_logger, get_model


def main(args):
    print("Args: ", args)
    logging = get_logger(args=args, task='i2t')

    config = get_config(dataset=args.dataset, model=args.model)

    logging.info(
        "Loading the annotations, preparing the dataset, the model, image embeddings..."
    )
    json_file = load_json_annotations(config=config)

    ds_test_split = Dataset(config=config, split="test", json_file=json_file)

    # load the model
    model = get_model(args.model, config=config)

    # load precomputed image embeddings
    capt_ids, capts, capt_embs = model.compute_caption_embeddings(
        ds_split=ds_test_split,
        compute_from_scratch=args.compute_from_scratch
    )

    rel_estimator = RelevanceEstimator(config=config, dataset=ds_test_split)
    retriever = Retriever(config=config, model=model)
    dcg = DCG(config=config, rel_estimator=rel_estimator)

    i2t_queries = []
    i2t_targets = []
    i2t_retrieved_documents = []
    i2t_scores = []
    i2t_recalls_at_1 = []
    i2t_recalls_at_5 = []
    i2t_recalls_at_10 = []
    i2t_dcgs = []

    logging.info("Image-to-text evaluation...")
    seen_image_ids = []
    for datapoint in tqdm(ds_test_split):
        # get textual query and target
        query = datapoint[1]
        target_filename = datapoint[3]

        if target_filename in seen_image_ids:
            continue
        seen_image_ids.append(target_filename)

        retrieved_caption_ids, scores = retriever.retrieve_top_k(
            query=query, documents=capt_embs, documents_names=capt_ids, k=10
        )

        associated_img_ids = [
            ds_test_split.captions[capt_id]["imgid"]
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

        i2t_dcg = dcg.compute_dcg(
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
            logging.info(f"Progress: {datapoint[-1]}/{len(ds_test_split)}")

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

    print(i2t_results.describe())
    print(i2t_results.head())

    logging.info("Saving the results...")
    save_results_dataframe(
        config=config, dataf=i2t_results, root=f'{args.model}/{args.dataset}', filename=f"i2t-results"
    )

    logging.info("Done!")


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
        choices=["clip", "blip", "flava", "beit", "xvlm"],
        help="model name: clip, blip, flava, beit",
    )
    parser.add_argument(
        "--compute_from_scratch",
        type=bool,
        default=False,
        help="Compute embeddings from scratch?",
    )
    args = parser.parse_args()
    main(args)
