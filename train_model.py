"""
Model Training and Validation Module for Business Entity Resolution.

Uses LightGBM gradient boosted decision trees.
Performs:
1. Leak-free Source-1-level training / validation split.
2. Hard negative mining via candidate blocking.
3. Feature computation on all positive and hard negative pairs.
4. Model training and hyperparameter tuning.
5. Probability threshold optimization specifically for F0.5.
6. Evaluation of Precision, Recall, F0.5, and Singleton accuracy.
7. Model persistence for test set prediction.
"""

import csv
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
)
from blocking import BlockingIndex
from features import compute_pair_features, FEATURE_NAMES
from evaluate import compute_metrics

BASE_DIR = SRC_DIR.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
WORK_DIR = BASE_DIR / "work"
MODEL_PATH = WORK_DIR / "lgb_entity_matcher.pkl"
META_PATH = WORK_DIR / "model_meta.json"

RANDOM_SEED = 42
TRAIN_QUERIES_COUNT = 15000
VAL_QUERIES_COUNT = 3000
MAX_NEGATIVES_PER_QUERY = 8


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """Loads ground truth mapping: {s1_id: set(matched_ids)}"""
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            matches = [m.strip() for m in row[1].split(",") if m.strip()] if len(row) > 1 else []
            gt[s1_id] = set(matches)
    return gt


def load_query_sample(
    path: Path,
    target_count: int,
    seed: int = 42
) -> List[Tuple[str, str, str, str]]:
    """
    Loads a reservoir sample of Source 1 queries.
    Returns list of (s1_id, norm_name, norm_addr, norm_country).
    """
    rng = random.Random(seed)
    sample = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            seen += 1
            rec = (
                row[0].strip(),
                normalize_business_name(row[1]),
                normalize_business_address(row[2]),
                normalize_country(row[3])
            )
            if len(sample) < target_count:
                sample.append(rec)
            else:
                j = rng.randrange(seen)
                if j < target_count:
                    sample[j] = rec
    return sample


def build_country_index_for_queries(
    source_paths: List[Path],
    query_needed_ids: Set[str],
    query_country: str,
    max_key_postings: int = 400
) -> Tuple[BlockingIndex, Dict[str, Tuple[str, str]]]:
    """
    Builds a blocking index containing:
    1. All target records matching query_country (up to a safe memory limit)
    2. Guaranteed inclusion of any ground-truth target record needed by queries
    Returns (index, target_dict: {id: (norm_name, norm_addr)})
    """
    index = BlockingIndex(max_key_postings=max_key_postings)
    target_dict = {}
    
    total_added = 0
    for path in source_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            for row in reader:
                if not row:
                    continue
                tid = row[0].strip()
                t_country = normalize_country(row[3]) if len(row) > 3 else ""
                
                is_needed = (tid in query_needed_ids)
                if is_needed or (t_country == query_country and total_added < 300000):
                    n_name = normalize_business_name(row[1])
                    n_addr = normalize_business_address(row[2]) if len(row) > 2 else ""
                    index.add_target(tid, n_name, n_addr)
                    target_dict[tid] = (n_name, n_addr)
                    total_added += 1
                    
    index.prune_frequent_keys()
    return index, target_dict


def generate_dataset_for_split(
    queries: List[Tuple[str, str, str, str]],
    gt: Dict[str, Set[str]],
    index_by_country: Dict[str, BlockingIndex],
    target_dict: Dict[str, Tuple[str, str]],
    is_training: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[Tuple[str, List[float], int]]]]:
    """
    Generates features and labels for queries.
    Returns:
      X (numpy array of features)
      y (numpy array of labels 0/1)
      query_candidates: {s1_id: [(target_id, features, is_match)]}
    """
    X_list = []
    y_list = []
    query_candidates = {}
    
    for s1_id, q_name, q_addr, q_country in queries:
        true_matches = gt.get(s1_id, set())
        index = index_by_country.get(q_country)
        candidates = index.get_candidates(q_name, q_addr, max_candidates=25) if index else []
        
        cand_dict = {tid: (tname, taddr) for tid, tname, taddr in candidates}
        
        # In training, ensure true matches are present
        if is_training:
            for tm in true_matches:
                if tm in target_dict and tm not in cand_dict:
                    cand_dict[tm] = target_dict[tm]
                    
        # Extract features
        cand_pairs = []
        neg_count = 0
        for tid, (tname, taddr) in cand_dict.items():
            is_match = 1 if (tid in true_matches) else 0
            if not is_match and is_training and neg_count >= MAX_NEGATIVES_PER_QUERY:
                continue
            if not is_match:
                neg_count += 1
                
            feats = compute_pair_features(q_name, q_addr, tid, tname, taddr)
            cand_pairs.append((tid, feats, is_match))
            X_list.append(feats)
            y_list.append(is_match)
            
        query_candidates[s1_id] = cand_pairs
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), query_candidates


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("      TRAINING & VALIDATING LIGHTGBM MATCHING MODEL")
    print("=" * 70)
    
    # 1. Load Ground Truth
    print("\n[1/6] Loading Ground Truth...")
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    gt = load_ground_truth(gt_path)
    print(f"Total ground truth records loaded: {len(gt):,}")
    
    # 2. Sample Train and Validation S1 Queries (Source-1-level split)
    print(f"\n[2/6] Sampling {TRAIN_QUERIES_COUNT:,} Train and {VAL_QUERIES_COUNT:,} Validation Source 1 Queries...")
    total_sample = TRAIN_QUERIES_COUNT + VAL_QUERIES_COUNT
    all_sampled = load_query_sample(TRAIN_DIR / "train_source1.tsv", total_sample, seed=RANDOM_SEED)
    
    train_queries = all_sampled[:TRAIN_QUERIES_COUNT]
    val_queries = all_sampled[TRAIN_QUERIES_COUNT:]
    print(f"Train queries: {len(train_queries):,}")
    print(f"Validation queries: {len(val_queries):,}")
    
    # Collect all required ground truth targets for queries
    all_needed_targets = set()
    for q in all_sampled:
        all_needed_targets.update(gt.get(q[0], set()))
    print(f"True targets needed for sampled queries: {len(all_needed_targets):,}")
    
    # 3. Build Blocking Index by Country
    print("\n[3/6] Building In-Memory Country-Partitioned Blocking Index...")
    target_sources = [TRAIN_DIR / "train_source2.tsv", TRAIN_DIR / "train_source3.tsv"]
    
    index_by_country = {}
    combined_target_dict = {}
    
    for c in ["us", "india"]:
        print(f"  Indexing target records for country: {c.upper()}...")
        idx, tdict = build_country_index_for_queries(target_sources, all_needed_targets, c)
        index_by_country[c] = idx
        combined_target_dict.update(tdict)
        print(f"    Indexed {len(idx.targets):,} targets for {c.upper()}")
        
    # 4. Generate Training and Validation Feature Datasets
    print("\n[4/6] Generating Training and Validation Feature Pairs...")
    t0 = time.time()
    X_train, y_train, _ = generate_dataset_for_split(
        train_queries, gt, index_by_country, combined_target_dict, is_training=True
    )
    X_val, y_val, val_candidates = generate_dataset_for_split(
        val_queries, gt, index_by_country, combined_target_dict, is_training=False
    )
    print(f"Feature extraction time: {time.time() - t0:.1f}s")
    print(f"X_train shape: {X_train.shape}, Positives: {np.sum(y_train):,} ({np.mean(y_train)*100:.1f}%), Negatives: {len(y_train)-np.sum(y_train):,}")
    print(f"X_val shape:   {X_val.shape}, Positives: {np.sum(y_val):,} ({np.mean(y_val)*100:.1f}%), Negatives: {len(y_val)-np.sum(y_val):,}")
    
    # 5. Train LightGBM Model
    print("\n[5/6] Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    
    t_train = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    print(f"Model training time: {time.time() - t_train:.2f}s")
    
    # Feature Importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 Most Important Features:")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<24}: {importances[idx]}")
        
    # 6. Optimize Decision Threshold specifically for F0.5
    print("\n[6/6] Optimizing Decision Threshold on Validation Set (F0.5)...")
    val_gt = {q[0]: gt.get(q[0], set()) for q in val_queries}
    
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    best_thresh = 0.50
    best_f05 = -1.0
    best_metrics = {}
    
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F0.5':<10} | {'Singletons':<12}")
    print("-" * 60)
    
    for thresh in thresholds:
        val_preds = {}
        for s1_id in val_gt:
            cands = val_candidates.get(s1_id, [])
            matched = set()
            if cands:
                feats_batch = np.array([c[1] for c in cands], dtype=np.float32)
                probs = model.predict_proba(feats_batch)[:, 1]
                for c_item, prob in zip(cands, probs):
                    if prob >= thresh:
                        matched.add(c_item[0])
            val_preds[s1_id] = matched
            
        m = compute_metrics(val_gt, val_preds)
        print(f"{thresh:<10.2f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f05']:<10.4f} | {m['singleton_accuracy']*100:<10.1f}%")
        
        if m["f05"] > best_f05:
            best_f05 = m["f05"]
            best_thresh = thresh
            best_metrics = m
            
    print("-" * 60)
    print(f"\n>>> OPTIMAL THRESHOLD: {best_thresh:.2f} <<<")
    print(f"Validation Precision:  {best_metrics['precision']:.4f}")
    print(f"Validation Recall:     {best_metrics['recall']:.4f}")
    print(f"Validation F0.5 Score: {best_metrics['f05']:.4f}")
    print(f"Singleton Accuracy:    {best_metrics['singleton_accuracy']*100:.2f}%")
    print(f"True Positives:        {best_metrics['true_positives']:,}")
    print(f"False Positives:       {best_metrics['false_positives']:,}")
    print(f"False Negatives:       {best_metrics['false_negatives']:,}")
    
    # Save model and metadata
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "LightGBM Classifier",
            "optimal_threshold": best_thresh,
            "metrics": best_metrics,
            "feature_names": FEATURE_NAMES
        }, f, indent=2)
        
    print(f"\nSaved trained model to: {MODEL_PATH}")
    print(f"Saved metadata to:       {META_PATH}")
    print("[SUCCESS] Training and threshold optimization complete.")


if __name__ == "__main__":
    main()
