"""
Feature Engineering Module for Business Entity Resolution.

Computes comprehensive numerical similarity features for candidate pairs:
- String edit distances & token similarities (name & address)
- Exact match indicators
- Token Jaccard similarities & overlap counts
- Postal code and street number agreement
- Combined composite similarity metrics
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

from rapidfuzz import fuzz

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalization import extract_postal_codes

NUM_REGEX = re.compile(r"\b\d{1,6}[a-z]?\b", flags=re.IGNORECASE)

FEATURE_NAMES = [
    "name_exact",
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "name_common_tokens",
    "name_len_diff",
    "name_len_ratio",
    "name_prefix4_match",
    "addr_is_empty",
    "addr_exact",
    "addr_ratio",
    "addr_partial_ratio",
    "addr_token_set_ratio",
    "addr_jaccard",
    "addr_common_tokens",
    "addr_len_diff",
    "postal_match",
    "number_overlap",
    "combined_score",
    "is_source2",
]


def compute_pair_features(
    q_name: str,
    q_addr: str,
    t_id: str,
    t_name: str,
    t_addr: str
) -> List[float]:
    """
    Computes numerical feature vector for query (q_name, q_addr) and target (t_name, t_addr).
    """
    # 1. Name Features
    name_exact = 1.0 if q_name and q_name == t_name else 0.0
    name_ratio = fuzz.ratio(q_name, t_name) / 100.0 if q_name and t_name else 0.0
    name_partial_ratio = fuzz.partial_ratio(q_name, t_name) / 100.0 if q_name and t_name else 0.0
    name_token_sort_ratio = fuzz.token_sort_ratio(q_name, t_name) / 100.0 if q_name and t_name else 0.0
    name_token_set_ratio = fuzz.token_set_ratio(q_name, t_name) / 100.0 if q_name and t_name else 0.0
    
    q_tokens = set(q_name.split()) if q_name else set()
    t_tokens = set(t_name.split()) if t_name else set()
    inter_len = len(q_tokens & t_tokens)
    union_len = len(q_tokens | t_tokens)
    name_jaccard = (inter_len / union_len) if union_len > 0 else 0.0
    name_common_tokens = float(inter_len)
    
    lq = len(q_name)
    lt = len(t_name)
    name_len_diff = float(abs(lq - lt))
    name_len_ratio = (min(lq, lt) / max(lq, lt)) if max(lq, lt) > 0 else 0.0
    
    name_prefix4 = 1.0 if (len(q_name) >= 4 and len(t_name) >= 4 and q_name[:4] == t_name[:4]) else 0.0
    
    # 2. Address Features
    addr_is_empty = 1.0 if (not q_addr or not t_addr) else 0.0
    if addr_is_empty == 1.0:
        addr_exact = 0.0
        addr_ratio = 0.0
        addr_partial_ratio = 0.0
        addr_token_set_ratio = 0.0
        addr_jaccard = 0.0
        addr_common_tokens = 0.0
        addr_len_diff = float(abs(len(q_addr) - len(t_addr)))
        postal_match = 0.0
        number_overlap = 0.0
    else:
        addr_exact = 1.0 if q_addr == t_addr else 0.0
        addr_ratio = fuzz.ratio(q_addr, t_addr) / 100.0
        addr_partial_ratio = fuzz.partial_ratio(q_addr, t_addr) / 100.0
        addr_token_set_ratio = fuzz.token_set_ratio(q_addr, t_addr) / 100.0
        
        qa_tokens = set(q_addr.split())
        ta_tokens = set(t_addr.split())
        ainter = len(qa_tokens & ta_tokens)
        aunion = len(qa_tokens | ta_tokens)
        addr_jaccard = (ainter / aunion) if aunion > 0 else 0.0
        addr_common_tokens = float(ainter)
        addr_len_diff = float(abs(len(q_addr) - len(t_addr)))
        
        # Postal match
        qp = extract_postal_codes(q_addr)
        tp = extract_postal_codes(t_addr)
        if qp and tp:
            postal_match = 1.0 if set(qp) & set(tp) else -1.0
        else:
            postal_match = 0.0
            
        # Number overlap
        qn = set(NUM_REGEX.findall(q_addr))
        tn = set(NUM_REGEX.findall(t_addr))
        number_overlap = 1.0 if (qn and tn and (qn & tn)) else 0.0

    # 3. Composite & Target Source Features
    combined_score = (
        0.40 * name_token_set_ratio
        + 0.20 * name_ratio
        + 0.25 * addr_token_set_ratio
        + 0.15 * addr_ratio
    )
    is_source2 = 1.0 if t_id.startswith("S2") else 0.0

    return [
        name_exact,
        name_ratio,
        name_partial_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_jaccard,
        name_common_tokens,
        name_len_diff,
        name_len_ratio,
        name_prefix4,
        addr_is_empty,
        addr_exact,
        addr_ratio,
        addr_partial_ratio,
        addr_token_set_ratio,
        addr_jaccard,
        addr_common_tokens,
        addr_len_diff,
        postal_match,
        number_overlap,
        combined_score,
        is_source2,
    ]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== Testing Feature Engineering Module ===")
    
    feats = compute_pair_features(
        q_name="payne enterprises",
        q_addr="3315 fremont street peoria il",
        t_id="S2-478959098",
        t_name="payne enterprises",
        t_addr="3315 fremont st peoria il",
    )
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name:<24}: {val:.4f}")
        
    print(f"\nTotal Features: {len(FEATURE_NAMES)}")
    print("[SUCCESS] Feature module verified.")
