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
    logging = get_logger(args=args, task='t2i')

    config = get_config(dataset=args.dataset, model=args.model)

    logging.info(
        "Loading the annotations, preparing the dataset, the model, image embeddings..."
    )
    json_file = load_json_annotations(config=config)

    ds_test_split = Dataset(config=config, split="test", json_file=json_file)

    # load the model
    model = get_model(args.model, config=config)

    # load precomputed image embeddings
    img_filenames, img_emb = model.compute_image_embeddings(compute_from_scratch=True)
    # print('img_filenames: ', img_filenames[:10])

    rel_estimator = RelevanceEstimator(config=config, dataset=ds_test_split)
    retriever = Retriever(config=config, model=model)
    dcg = DCG(config=config, rel_estimator=rel_estimator)

    t2i_queries = []
    t2i_targets = []
    t2i_retrieved_documents = []
    t2i_scores = []
    t2i_recalls_at_1 = []
    t2i_recalls_at_5 = []
    t2i_recalls_at_10 = []
    t2i_dcgs = []

    logging.info("Text-to-image evaluation...")
    for datapoint in tqdm(ds_test_split):
        # get textual query and target
        query = datapoint[0]
        target_filename = datapoint[4]

        retrieved_documents, scores = retriever.retrieve_top_k(
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
        # print('t2i: recalls at 1, 5, 10: ', t2i_recall_at_1, t2i_recall_at_5, t2i_recall_at_10)

        t2i_dcg = dcg.compute_dcg(
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
            logging.info(f"Progress: {datapoint[-1]}/{len(ds_test_split)}")

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

    print(t2i_results.describe())
    print(t2i_results)

    logging.info("Saving the results...")
    save_results_dataframe(
        config=config, dataf=t2i_results, root=f'{args.model}/{args.dataset}', filename=f"t2i-results"
    )
    print("Done!")


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
    args = parser.parse_args()
    main(args)
