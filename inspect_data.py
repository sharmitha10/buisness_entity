import os
import sys
import csv
from collections import Counter
import numpy as np

# Ensure UTF-8 output for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

def inspect_source_file(file_path):
    print(f"\n=======================================================")
    print(f"File: {os.path.relpath(file_path, BASE_DIR)}")
    print(f"Absolute Path: {file_path}")
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")
    
    total_rows = 0
    missing_counts = Counter()
    countries = Counter()
    name_lens = []
    addr_lens = []
    entity_ids = set()
    duplicate_ids = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        cols = [c.strip().lstrip('\ufeff') for c in header]
        print(f"Columns: {cols}")
        
        # Sample for data types
        dtypes = {col: "string" for col in cols}
        print(f"Inferred Data Types: {dtypes}")
        
        for row in reader:
            if not row:
                continue
            total_rows += 1
            
            # Map columns
            row_dict = {cols[i]: row[i] if i < len(row) else "" for i in range(len(cols))}
            
            eid = row_dict.get("entity_id", "")
            if eid in entity_ids:
                duplicate_ids += 1
            else:
                entity_ids.add(eid)
                
            bname = row_dict.get("business_name", "")
            baddr = row_dict.get("business_address", "")
            country = row_dict.get("country", "")
            
            for c, val in row_dict.items():
                if not val or val.strip() == "":
                    missing_counts[c] += 1
                    
            if country:
                countries[country.strip()] += 1
            else:
                countries["<MISSING>"] += 1
                
            # Sample lengths for distribution stats (every 10th row to keep memory low)
            if total_rows % 10 == 0:
                name_lens.append(len(bname))
                addr_lens.append(len(baddr))
                
    print(f"Total Rows: {total_rows:,}")
    print(f"Unique Entity IDs: {len(entity_ids):,} (Duplicate IDs: {duplicate_ids})")
    print(f"Missing Values: {dict(missing_counts)}")
    print(f"Unique Countries ({len(countries)}): {dict(countries.most_common(10))}")
    
    if name_lens:
        print(f"Business Name Lengths: min={min(name_lens)}, mean={np.mean(name_lens):.1f}, median={np.median(name_lens):.1f}, max={max(name_lens)}")
    if addr_lens:
        print(f"Business Address Lengths: min={min(addr_lens)}, mean={np.mean(addr_lens):.1f}, median={np.median(addr_lens):.1f}, max={max(addr_lens)}")
        
    return {
        "rows": total_rows,
        "unique_ids": len(entity_ids),
        "countries": set(countries.keys())
    }

def inspect_ground_truth(file_path):
    print(f"\n=======================================================")
    print(f"File: {os.path.relpath(file_path, BASE_DIR)}")
    print(f"Absolute Path: {file_path}")
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")
    
    total_rows = 0
    match_counts = []
    source_counts = Counter()
    total_matches = 0
    s1_ids = set()
    duplicate_s1 = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        cols = [c.strip().lstrip('\ufeff') for c in header]
        print(f"Columns: {cols}")
        
        for row in reader:
            if not row:
                continue
            total_rows += 1
            s1_id = row[0].strip() if len(row) > 0 else ""
            matches_str = row[1].strip() if len(row) > 1 else ""
            
            if s1_id in s1_ids:
                duplicate_s1 += 1
            else:
                s1_ids.add(s1_id)
                
            if matches_str:
                matched_ids = [m.strip() for m in matches_str.split(",") if m.strip()]
            else:
                matched_ids = []
                
            num_matches = len(matched_ids)
            match_counts.append(num_matches)
            total_matches += num_matches
            
            for mid in matched_ids:
                if mid.startswith("S2"):
                    source_counts["Source 2"] += 1
                elif mid.startswith("S3"):
                    source_counts["Source 3"] += 1
                else:
                    source_counts["Other"] += 1
                    
    print(f"Total Source 1 Records with Ground Truth entry: {total_rows:,}")
    print(f"Unique Source 1 IDs in Ground Truth: {len(s1_ids):,} (Duplicate rows: {duplicate_s1})")
    print(f"Total True Matches: {total_matches:,}")
    print(f"Matches Breakdown by Target Source: {dict(source_counts)}")
    zero_matches = sum(1 for c in match_counts if c == 0)
    print(f"Source 1 Records with 0 matches (Singletons in GT): {zero_matches:,} ({zero_matches/total_rows*100:.2f}%)")
    print(f"Matches per Source 1 Entity: min={min(match_counts)}, mean={np.mean(match_counts):.2f}, median={np.median(match_counts):.1f}, max={max(match_counts)}")
    
    match_dist = Counter(match_counts)
    print(f"Distribution of match count per Source 1: {sorted(match_dist.items())[:10]} ...")
    
    return {
        "total_gt_s1": total_rows,
        "total_matches": total_matches,
        "singletons": zero_matches
    }

def main():
    print("=" * 60)
    print("      DATASET INSPECTION & SUMMARY REPORT")
    print("=" * 60)
    
    # 1. Training set
    print("\n>>> INSPECTING TRAINING FILES <<<")
    train_s1 = inspect_source_file(os.path.join(TRAIN_DIR, "train_source1.tsv"))
    train_s2 = inspect_source_file(os.path.join(TRAIN_DIR, "train_source2.tsv"))
    train_s3 = inspect_source_file(os.path.join(TRAIN_DIR, "train_source3.tsv"))
    gt = inspect_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
    
    # 2. Test set
    print("\n>>> INSPECTING TEST FILES <<<")
    test_s1 = inspect_source_file(os.path.join(TEST_DIR, "test_source1.tsv"))
    test_s2 = inspect_source_file(os.path.join(TEST_DIR, "test_source2.tsv"))
    test_s3 = inspect_source_file(os.path.join(TEST_DIR, "test_source3.tsv"))
    
    # Overall summary comparison
    print("\n" + "=" * 60)
    print("               CONCISE SUMMARY OVERVIEW")
    print("=" * 60)
    print(f"{'Split':<8} | {'Source 1':<12} | {'Source 2':<12} | {'Source 3':<12} | {'Total S2+S3':<12}")
    print("-" * 65)
    print(f"{'Train':<8} | {train_s1['rows']:<12,} | {train_s2['rows']:<12,} | {train_s3['rows']:<12,} | {train_s2['rows']+train_s3['rows']:<12,}")
    print(f"{'Test':<8} | {test_s1['rows']:<12,} | {test_s2['rows']:<12,} | {test_s3['rows']:<12,} | {test_s2['rows']+test_s3['rows']:<12,}")
    print("-" * 65)
    print(f"Ground Truth entries: {gt['total_gt_s1']:,} S1 entities, {gt['total_matches']:,} total true matches.")
    print("=" * 60)

if __name__ == "__main__":
    main()
