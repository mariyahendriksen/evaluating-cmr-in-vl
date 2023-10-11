import math
import torch.nn as nn
from typing import Type, List
import PIL
import inspect


class DCG():

    def __init__(self, config, rel_estimator) -> None:

        super(DCG, self).__init__()

        self.config = config

        self.rel_estimator = rel_estimator


    def get_gains(self, query: str, target_filename: str, retrieved_documents: List[str], caption_ids=None) -> List[int]:
        """Get list of binary matches given a target filename and top-k retrieved filenames

        Args:
            query (str): query used to compute the score
            target_filename (str): target filename, e.g., '2841449931.jpg'
            retrieved_documents (List[str]): list of retrieved filenames,

        Returns:
            List[int]: gain scores
        """
        gains = []
        for idx, doc in enumerate(retrieved_documents):
            if target_filename == doc:
                # print('Target_filename == doc')
                gain = 1
            elif target_filename != doc:
                # print('Target_filename != doc; computing cross-modal relevance score...')
                if isinstance(query, str):
                    # print('Query is a string')
                    # print('Gains: query: ', query)
                    # print('Gains: doc: ', doc)
                    gain = self.rel_estimator.compute_relevance_estimation_t2i(query=query, document=doc)
                elif isinstance(query, PIL.Image.Image):
                #     print('Query is an image ')
                #     print('Query: ', query)
                #     print('Document: ', doc)
                #     print('capt id: ', caption_ids[idx])
                #     print('I2T RELEVANCE ESTIMATION: ', inspect.getsource(self.rel_estimator.compute_relevance_estimation_i2t))
                    gain = self.rel_estimator.compute_relevance_estimation_i2t(query=query, document=doc, caption_id=caption_ids[idx])
                else:
                    raise NotImplementedError
            else:
                print('Problem with ', doc)
            # print('Gain: ', gain)
            gains.append(gain)

        assert len(gains) == len(retrieved_documents)

        return gains


    def gains_to_dcg(self, gains: List[float]) -> float:

        ranks = list(range(1, len(gains)+1))

        assert len(gains) == len(ranks)

        dcgs = []
        # print('Ranks: ', ranks)
        # print('Gains: ', gains)
        for rank, gain in zip(ranks, gains):
            # print(f'Rank: {rank}; gain: {rank}')
            # print(f'Types : Rank: {type(rank)}; gain: {type(rank)}')
            # TODO: make it numpy
            dcg = gain/(math.log(rank + 1, 2))
            dcgs.append(dcg)
        return round(sum(dcgs), 4)


    def compute_dcg(self, query, target_filename, retrieved_documents, caption_ids=None) -> float:

        gains = self.get_gains(
            query=query,
            target_filename=target_filename,
            retrieved_documents=retrieved_documents,
            caption_ids=caption_ids
            )
        dcg_score = self.gains_to_dcg(gains=gains)
        return dcg_score
