"""
Prediction Module for Business Entity Resolution.

Runs the full pipeline on the test set:
1. Loads the trained LightGBM model and optimal threshold.
2. Discovers all countries in test_source1.tsv (open-set — includes France).
3. Builds a per-country BlockingIndex from test_source2.tsv and test_source3.tsv.
4. For every test Source-1 entity:
   a. Retrieves candidates via blocking index.
   b. Computes 22 similarity features for every candidate pair.
   c. Scores with LightGBM.
   d. Applies the optimal threshold.
5. Writes output/matching_results.tsv and output/candidate_pairs.tsv.
6. Guarantees every S1 test entity appears exactly once (singletons get an empty matched_entity_ids).
"""

import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

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
from features import compute_pair_features

BASE_DIR   = SRC_DIR.parent
DATASET_DIR = BASE_DIR / "dataset"
TEST_DIR   = DATASET_DIR / "test"
WORK_DIR   = BASE_DIR / "work"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_PATH = WORK_DIR / "lgb_entity_matcher.pkl"
META_PATH  = WORK_DIR / "model_meta.json"

MAX_CANDIDATES = 30


def load_model_and_threshold():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = meta["optimal_threshold"]
    print(f"  Model        : {meta['model_type']}")
    print(f"  Threshold    : {threshold}")
    print(f"  Val F0.5     : {meta['metrics']['f05']:.4f}")
    print(f"  Val Precision: {meta['metrics']['precision']:.4f}")
    print(f"  Val Recall   : {meta['metrics']['recall']:.4f}")
    return model, threshold


def discover_countries(path: Path) -> dict:
    counts = defaultdict(int)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row:
                counts[normalize_country(row[3] if len(row) > 3 else "")] += 1
    return dict(counts)


def build_index_for_country(source_paths, target_country, max_key_postings=500):
    index = BlockingIndex(max_key_postings=max_key_postings)
    target_dict = {}
    for path in source_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if not row:
                    continue
                tid = row[0].strip()
                t_country = normalize_country(row[3] if len(row) > 3 else "")
                if t_country == target_country:
                    n_name = normalize_business_name(row[1])
                    n_addr = normalize_business_address(row[2] if len(row) > 2 else "")
                    index.add_target(tid, n_name, n_addr)
                    target_dict[tid] = (n_name, n_addr)
    index.prune_frequent_keys()
    return index, target_dict


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 70)
    print("  PREDICTING ON TEST SET")
    print("=" * 70)

    # 1. Load model
    print("\n[1/5] Loading trained LightGBM model...")
    model, threshold = load_model_and_threshold()

    # 2. Discover countries in test S1
    print("\n[2/5] Discovering countries in test_source1.tsv...")
    s1_countries = discover_countries(TEST_DIR / "test_source1.tsv")
    print(f"  Countries found: {s1_countries}")

    # 3. Build per-country blocking index
    print("\n[3/5] Building per-country blocking indices for test S2/S3...")
    target_sources = [TEST_DIR / "test_source2.tsv", TEST_DIR / "test_source3.tsv"]
    index_by_country = {}
    target_dict_by_country = {}

    for country in sorted(s1_countries.keys()):
        if not country:
            continue
        print(f"  Indexing targets for country: '{country}' ...")
        t0 = time.time()
        idx, tdict = build_index_for_country(target_sources, country)
        index_by_country[country] = idx
        target_dict_by_country[country] = tdict
        print(f"    -> {len(idx.targets):,} targets indexed in {time.time()-t0:.1f}s")
        print(f"    -> Unique index keys: {len(idx.index):,}")

    # 4. Score every S1 test entity
    print("\n[4/5] Scoring candidates for every test Source-1 entity...")
    matching_out  = {}
    candidate_out = {}
    total_s1 = 0
    total_cands = 0
    total_matches = 0
    singletons = 0
    log_every = 100_000
    t0 = time.time()

    with open(TEST_DIR / "test_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if not row:
                continue
            s1_id     = row[0].strip()
            q_name    = normalize_business_name(row[1])
            q_addr    = normalize_business_address(row[2] if len(row) > 2 else "")
            q_country = normalize_country(row[3] if len(row) > 3 else "")

            total_s1 += 1

            idx = index_by_country.get(q_country)
            candidates = idx.get_candidates(q_name, q_addr, max_candidates=MAX_CANDIDATES) if idx else []

            cand_ids = {tid for tid, _, _ in candidates}
            candidate_out[s1_id] = cand_ids
            total_cands += len(cand_ids)

            if not candidates:
                matching_out[s1_id] = set()
                singletons += 1
            else:
                feats = np.array(
                    [compute_pair_features(q_name, q_addr, tid, tname, taddr)
                     for tid, tname, taddr in candidates],
                    dtype=np.float32
                )
                probs = model.predict_proba(feats)[:, 1]
                matched = {candidates[i][0] for i, p in enumerate(probs) if p >= threshold}
                matching_out[s1_id] = matched
                if matched:
                    total_matches += len(matched)
                else:
                    singletons += 1

            if total_s1 % log_every == 0:
                elapsed = time.time() - t0
                print(f"  Processed {total_s1:,} | matches: {total_matches:,} "
                      f"| speed: {total_s1/elapsed:,.0f}/s")

    elapsed = time.time() - t0
    print(f"\n  Done: {total_s1:,} S1 entities in {elapsed:.1f}s  ({total_s1/elapsed:,.0f}/s)")
    print(f"  Total candidates generated    : {total_cands:,}")
    print(f"  Avg candidates per S1 entity  : {total_cands/total_s1:.1f}")
    print(f"  Total matched pairs produced  : {total_matches:,}")
    print(f"  Predicted singletons          : {singletons:,}  ({singletons/total_s1*100:.2f}%)")

    # 5. Write output files in original S1 order
    print("\n[5/5] Writing output files...")
    match_path = OUTPUT_DIR / "matching_results.tsv"
    cand_path  = OUTPUT_DIR / "candidate_pairs.tsv"

    with open(match_path, "w", encoding="utf-8", newline="") as mf, \
         open(cand_path,  "w", encoding="utf-8", newline="") as cf:

        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")

        with open(TEST_DIR / "test_source1.tsv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if not row:
                    continue
                s1_id    = row[0].strip()
                matched  = matching_out.get(s1_id, set())
                cand_ids = candidate_out.get(s1_id, set())
                mf.write(f"{s1_id}\t{','.join(sorted(matched))}\n")
                cf.write(f"{s1_id}\t{','.join(sorted(cand_ids))}\n")

    print(f"  Wrote: {match_path}")
    print(f"  Wrote: {cand_path}")
    print(f"\nTotal pipeline time: {(time.time()-t_start)/60:.2f} minutes")
    print("[SUCCESS] Prediction complete.")


if __name__ == "__main__":
    main()
