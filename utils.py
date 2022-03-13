from typing import List
import os
import json
import pandas as pd
import itertools
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import tensorflow as tf
#from keras.backend import set_session
import numpy as np
import random as rn
from sklearn.decomposition import PCA

SEED = 123
#tf.random.set_random_seed(SEED)
#tf.random.set_seed(SEED)

def get_score(cm, mode='test'):
    fs = []
    ps = []
    rs = []
    n_class = cm.shape[0]
    correct = []
    total = []
    for idx in range(n_class):
        TP = cm[idx][idx]
        correct.append(TP)
        total.append(cm[idx].sum())
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        fs.append(f * 100)
        ps.append(p * 100)
        rs.append(r * 100)

    f = np.mean(fs).round(2)
    p_seen = np.mean(ps[:-1]).round(2)
    r_seen = np.mean(rs[:-1]).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    p_unseen = round(ps[-1], 2)
    r_unseen = round(rs[-1], 2)
    f_unseen = round(fs[-1], 2)
    acc = (sum(correct) / sum(total) * 100).round(2)
    acc_in = (sum(correct[:-1]) / sum(total[:-1]) * 100).round(2)
    acc_ood = (correct[-1] / total[-1] * 100).round(2)
    if mode == 'test':
        print(f"Overall(macro): , f:{f}, acc:{acc}")
        print(f"Seen(macro): , f:{f_seen}, acc:{acc_in}, p:{p_seen}, r:{r_seen}")
        print(f"=====> Unseen(Experiment) <=====: , f:{f_unseen}, acc:{acc_ood}, p:{p_unseen}, r:{r_unseen}")

    return f, acc, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen


def get_test_info(texts: pd.Series,
                  label: pd.Series,
                  label_mask: pd.Series,
                  softmax_prob: np.ndarray,
                  softmax_classes: List[str],
                  lof_result: np.ndarray = None,
                  gda_result: np.ndarray = None,
                  gda_classes: List[str] = None,
                  save_to_file: bool = False,
                  output_dir: str = None) -> pd.DataFrame:
    """
    Return a pd.DataFrame, including the following information for each test instances:
        - the text of the instance
        - label & masked label of the sentence
        - the softmax probability for each seen classes (sum up to 1)
        - the softmax prediction
        - the softmax confidence (i.e. the max softmax probability among all seen classes)
        - (if use lof) lof prediction result (1 for in-domain and -1 for out-of-domain)
        - (if use gda) gda mahalanobis distance for each seen classes
        - (if use gda) the gda confidence (i.e. the min mahalanobis distance among all seen classes)
    """
    df = pd.DataFrame()
    df['label'] = label
    df['label_mask'] = label_mask
    for idx, _class in enumerate(softmax_classes):
        df[f'softmax_prob_{_class}'] = softmax_prob[:, idx]
    df['softmax_prediction'] = [softmax_classes[idx] for idx in softmax_prob.argmax(axis=-1)]
    df['softmax_confidence'] = softmax_prob.max(axis=-1)
    if lof_result is not None:
        df['lof_prediction'] = lof_result
    if gda_result is not None:
        for idx, _class in enumerate(gda_classes):
            df[f'm_dist_{_class}'] = gda_result[:, idx]
        df['gda_prediction'] = [gda_classes[idx] for idx in gda_result.argmin(axis=-1)]
        df['gda_confidence'] = gda_result.min(axis=-1)
    df['text'] = [text for text in texts]

    if save_to_file:
        df.to_csv(os.path.join(output_dir, "test_info.csv"))

    return df


def confidence(features: np.ndarray,
               means: np.ndarray,
               distance_type: str,
               cov: np.ndarray = None) -> np.ndarray:
    """
    Calculate mahalanobis or euclidean based confidence score for each class.

    Params:
        - features: shape (num_samples, num_features)
        - means: shape (num_classes, num_features)
        - cov: shape (num_features, num_features) or None (if use euclidean distance)

    Returns:
        - confidence: shape (num_samples, num_classes)
    """
    assert distance_type in ("euclidean", "mahalanobis")

    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = means.shape[0]
    if distance_type == "euclidean":
        cov = np.identity(num_features)

    features = features.reshape(num_samples, 1, num_features).repeat(num_classes,
                                                                     axis=1)  # (num_samples, num_classes, num_features)
    means = means.reshape(1, num_classes, num_features).repeat(num_samples,
                                                               axis=0)  # (num_samples, num_classes, num_features)
    vectors = features - means  # (num_samples, num_classes, num_features)
    cov_inv = np.linalg.inv(cov)
    bef_sqrt = np.matmul(np.matmul(vectors.reshape(num_samples, num_classes, 1, num_features), cov_inv),
                         vectors.reshape(num_samples, num_classes, num_features, 1)).squeeze()
    result = np.sqrt(bef_sqrt)
    result[np.isnan(result)] = 1e12  # solve nan
    return result

def estimate_best_threshold(seen_m_dist: np.ndarray,
                            unseen_m_dist: np.ndarray) -> float:
    """
    Given mahalanobis distance for seen and unseen instances in valid set, estimate
    a best threshold (i.e. achieving best f1 in valid set) for test set.
    """
    lst = []
    for item in seen_m_dist:
        lst.append((item, "seen"))
    for item in unseen_m_dist:
        lst.append((item, "unseen"))
    # sort by m_dist: [(5.65, 'seen'), (8.33, 'seen'), ..., (854.3, 'unseen')]
    lst = sorted(lst, key=lambda item: item[0])

    threshold = 0.
    tp, fp, fn = len(unseen_m_dist), len(seen_m_dist), 0

    def compute_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        return (2 * p * r) / (p + r + 1e-10)

    f1 = compute_f1(tp, fp, fn)

    for m_dist, label in lst:
        if label == "seen":  # fp -> tn
            fp -= 1
        else:  # tp -> fn
            tp -= 1
            fn += 1
        if compute_f1(tp, fp, fn) > f1:
            f1 = compute_f1(tp, fp, fn)
            threshold = m_dist + 1e-10

    print("estimated threshold:", threshold)
    return threshold

def plot_oos_ind(ind_val, oos_val):
    x_ind_val = np.array(range(len(ind_val)))
    x_oos_val = np.array(range(len(oos_val)))

    plt.scatter(x_ind_val, np.array(ind_val), c='green')
    plt.scatter(x_oos_val, np.array(oos_val), c='red')

    plt.show()
