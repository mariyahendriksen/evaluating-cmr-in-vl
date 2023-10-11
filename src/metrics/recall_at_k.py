from typing import List

def recall_at_k(target_filename: str, retrieved_documents: List[int], k=5, total_in_collection=5) -> float:
    """Compute recall at k

    Args:
        target (str): filename of the target image associated with the right answer
        retrieved_documents (List[int]): list of retrieved documents
        k (int, optional): cut off point for computing recall. Defaults to 5.
        total_in_collection (int, optional): total number of relevant items in the collection (for captions = 5, for images = 1). Defaults to 5.

    Returns:
        float: recall value
    """
    k_items = retrieved_documents[:k]
    return k_items.count(target_filename) / min(total_in_collection, len(k_items))
