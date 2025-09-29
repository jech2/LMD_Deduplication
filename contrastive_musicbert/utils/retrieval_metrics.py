### Code references ###
# nDCG
# https://gist.github.com/mblondel/7337391

import numpy as np
from sklearn.metrics import average_precision_score as ap

def find_first_one_np(binary_list):
    indices = np.where(np.array(binary_list) == 1)[0]
    return indices[0] if indices.size > 0 else None

# only for binary data
def mrr(y_pred, cutoff=None):
    if cutoff is not None:
        y_pred = y_pred[:cutoff]

    # rank of max predicted score
    rank_max = find_first_one_np(y_pred)
    
    if rank_max is not None:
        return 1./(rank_max+1)
    else:
        return 0.

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    # print('best', best)
    actual = dcg_score(y_true, y_score, k, gains)
    # print('actual', actual)
    return actual / best

def create_binary_list(duplicate_list, retrieval_result_list):
    total_len = len(retrieval_result_list)

    correct_answer_list = [1] * len(duplicate_list)
    correct_answer_list += [0] * (total_len - len(duplicate_list))
    retrieval_binary_list = [1 if result in duplicate_list else 0 for result in retrieval_result_list]
    
    return correct_answer_list, retrieval_binary_list

if __name__ == '__main__':
    # Example usage
    k = 5 
    duplicate_list = ['file1', 'file3', 'file5']
    retrieval_result_list = ['file1', 'file2', 'file3', 'file4', 'file5', 'file6', 'file7', 'file8', 'file9', 'file10']
    retrieval_similarity_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    ideal_retrieval_similarities = [1.0, 0.5, 0.9, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5]

    correct_answer_list, retrieval_binary_list = create_binary_list(duplicate_list, retrieval_result_list)

    print('#### RETRIEVAL EXAMPLE ####')
    print(f'k={k}')
    print(f'duplicate_file_list={duplicate_list}')
    print(f'retrieval_result_list={retrieval_result_list}')
    
    print("Correct Answer List (Binary):", correct_answer_list)
    print("Retrieval Result List (Binary):", retrieval_binary_list)

    ### REFERENCE ###
    print('\n#### REFERENCE SCORE FOR THE GROUND TRUTH ####')
    print('MRR', mrr(correct_answer_list))
    print('AP', ap(retrieval_binary_list, ideal_retrieval_similarities))
    print('R@1', recall_at_k(correct_answer_list, correct_answer_list, k=1))    
    print('R@3', recall_at_k(correct_answer_list, correct_answer_list, k=3))    
    print('R@5', recall_at_k(correct_answer_list, correct_answer_list, k=5))
    print('R@10', recall_at_k(correct_answer_list, correct_answer_list, k=10))
    print('nDCG@1', ndcg_score(retrieval_binary_list, ideal_retrieval_similarities, k=1))
    print('nDCG@3', ndcg_score(retrieval_binary_list, ideal_retrieval_similarities, k=3))
    print('nDCG@5', ndcg_score(retrieval_binary_list, ideal_retrieval_similarities, k=5))
    print('nDCG_all', ndcg_score(retrieval_binary_list, ideal_retrieval_similarities))
    print('nDCG_all', calculate_ndcg_all(0, duplicate_list, retrieval_similarity_list))
    

    ### RETRIEVAL ###
    print('\n#### RETRIEVAL SCORE ####')
    print('MRR', mrr(retrieval_binary_list))
    print('AP', ap(retrieval_binary_list, retrieval_similarity_list))
    print('R@1', recall_at_k(correct_answer_list, retrieval_binary_list, k=1))    
    print('R@3', recall_at_k(correct_answer_list, retrieval_binary_list, k=3))    
    print('R@5', recall_at_k(correct_answer_list, retrieval_binary_list, k=5))
    print('R@10', recall_at_k(correct_answer_list, retrieval_binary_list, k=10))
    print('nDCG@1', ndcg_score(retrieval_binary_list, retrieval_similarity_list, k=1))
    print('nDCG@3', ndcg_score(retrieval_binary_list, retrieval_similarity_list, k=3))
    print('nDCG@5', ndcg_score(retrieval_binary_list, retrieval_similarity_list, k=5))
    print('nDCG_all', ndcg_score(retrieval_binary_list, retrieval_similarity_list))