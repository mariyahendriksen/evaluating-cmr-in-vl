from typing import List
import torch.nn as nn
import os
from sentence_transformers import util
from IPython.display import display
from IPython.display import Image as IPImage


class Retriever(nn.Module):

    def __init__(self, config, model) -> None:

        super(Retriever, self).__init__()

        self.config = config

        self.model = model


    def retrieve_top_k(self, query, documents, documents_names, k=3, show_images=False) -> (List[str], List[float]):
        """Retrieve top k items from documents given a query

        Args:
            query (str|image): query
            documents (torch.Tensor): tensor of shape (n, m) where n - number of documents, m - embedding size
            documents_names (List[str]): list of size n, where n - number of documents; mapping between document embeddings and document names
            k (int, optional): length of the ranked list . Defaults to 3.
        Returns:
            doc_names (List[str]): retrieved document names
            doc_scores List[float]: scores of the retrieved documents
        """
        doc_names = []
        doc_scores = []

        if isinstance(query, str):
            query = query[:self.config.model.max_seq_length]

        # encode the query
        query_emb = self.model.encode(
            [query], convert_to_tensor=True
            # show_progress_bar=False
            )

        # get top-k hits
        # print('query emb: ', query_emb.shape)
        # print('documents shape: ', documents.shape)
        hits = util.semantic_search(query_emb, documents, top_k=k)[0]

        # print("Query:")
        # display(query)
        for hit in hits:
            doc_name, doc_score = documents_names[hit["corpus_id"]], round(hit["score"], 4)
            # print(f'Document: {doc_name}, score: {doc_score}')
            # print('hit: ', hit)
            if isinstance(query, str) and show_images:
                display(IPImage(os.path.join(
                    self.model.config.dataset.root,
                    self.model.config.dataset.img_folder,
                    documents_names[hit['corpus_id']]), width=200
                    ))
            # else:
            #     print(documents_names[hit['corpus_id']])
            doc_names.append(doc_name)
            doc_scores.append(doc_score)

        return doc_names, doc_scores
