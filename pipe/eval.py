import numpy as np


# ============================================================
# MAP@K (SPARSE SAFE)
# ============================================================

def compute_map_at_k_als(scores, test_matrix, k=10):
    n_users = test_matrix.shape[0]
    ap_scores = []

    for u in range(n_users):
        relevant = test_matrix[u].indices  # ✅ correct sparse access
        if len(relevant) == 0:
            continue

        top_k = np.argpartition(-scores[u], k)[:k]
        top_k = top_k[np.argsort(-scores[u][top_k])]

        hits = 0
        score = 0.0

        for rank, item in enumerate(top_k, start=1):
            if item in relevant:
                hits += 1
                score += hits / rank

        ap_scores.append(score / min(len(relevant), k))

    return float(np.mean(ap_scores)) if ap_scores else 0.0



def compute_map_at_k_cf(predictions, test_matrix, k=10):
    n_users = test_matrix.shape[0]
    ap_scores = []

    for user_idx in range(n_users):
        relevant_items = np.where(test_matrix[user_idx] > 0)[0]

        if len(relevant_items) == 0:
            continue

        # Top-k predictions
        top_k_items = np.argsort(predictions[user_idx])[::-1][:k]

        hits = 0
        score = 0.0

        for rank, item_idx in enumerate(top_k_items, start=1):
            if item_idx in relevant_items:
                hits += 1
                score += hits / rank

        ap_scores.append(score / min(len(relevant_items), k))

    return np.mean(ap_scores) if ap_scores else 0.0