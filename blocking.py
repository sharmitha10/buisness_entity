"""
Candidate Generation and Multi-Stage Blocking Module.

Efficiently filters the Cartesian product down to a high-recall, manageable candidate set:
1. Exact Country partitioning (never compares cross-country).
2. Multi-key Inverted Index:
   - Name prefix keys (4-char, 6-char)
   - Significant name tokens
   - Postal/PIN codes
   - Street/door numbers
   - Distinctive address tokens
3. High-frequency key pruning (avoids mega-clusters of common tokens).
4. Candidate ranking and top-K selection.
"""

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalization import (
    normalize_business_name,
    normalize_business_address,
    extract_postal_codes,
)

NAME_STOPWORDS = {
    "private", "limited", "corporation", "incorporated", "company",
    "limited liability company", "llc", "ltd", "inc", "corp", "pvt",
    "the", "and", "services", "service", "solutions", "solution",
    "technologies", "technology", "enterprises", "enterprise", "group",
    "holdings", "holding", "industries", "industry", "associates",
    "societe", "responsabilite", "limitee", "actions", "simplifiee"
}

ADDR_STOPWORDS = {
    "road", "street", "avenue", "lane", "drive", "boulevard", "court", "place",
    "floor", "building", "room", "number", "suite", "apartment", "unit",
    "phase", "sector", "block", "near", "behind", "opp", "opposite", "beside",
    "north", "south", "east", "west", "city", "town", "village",
    "district", "taluk", "delhi", "new", "india", "us", "usa", "france",
    "state", "st", "rd", "dr", "ave", "blvd", "box", "po"
}

COMPACT_REGEX = re.compile(r"[^\w]", flags=re.UNICODE)
NUM_REGEX = re.compile(r"\b\d{1,6}[a-z]?\b", flags=re.IGNORECASE)


def get_name_tokens(name: str) -> List[str]:
    return [t for t in name.split() if len(t) >= 3 and t not in NAME_STOPWORDS]


def get_addr_tokens(addr: str) -> List[str]:
    return [t for t in addr.split() if len(t) >= 3 and t not in ADDR_STOPWORDS]


def generate_blocking_keys(name: str, addr: str) -> Set[Tuple[str, str]]:
    """
    Generates blocking keys for a record given its normalized name and address.
    """
    keys = set()
    
    # 1. Name prefix keys
    compact = COMPACT_REGEX.sub("", name)
    if len(compact) >= 4:
        keys.add(("n4", compact[:4]))
    if len(compact) >= 6:
        keys.add(("n6", compact[:6]))
        
    # 2. Significant name tokens
    name_tokens = get_name_tokens(name)
    if name_tokens:
        keys.add(("nt0", name_tokens[0]))
        if len(name_tokens) > 1:
            keys.add(("nt1", name_tokens[1]))
            # Also add longest token if distinct
            longest = max(name_tokens, key=len)
            if len(longest) >= 5:
                keys.add(("nt_max", longest))
                
    # 3. Postal / PIN code
    postals = extract_postal_codes(addr)
    for p in postals:
        keys.add(("post", p))
        
    # 4. Street / Door numbers (from address)
    nums = NUM_REGEX.findall(addr)
    for n in nums[:2]:
        if len(n) >= 2:
            keys.add(("num", n.lower()))
            
    # 5. Distinctive address tokens
    addr_tokens = get_addr_tokens(addr)
    for at in addr_tokens[:2]:
        keys.add(("at", at))
        
    return keys


class BlockingIndex:
    """
    In-memory inverted index for target records belonging to a single country.
    """
    def __init__(self, max_key_postings: int = 500):
        self.max_key_postings = max_key_postings
        self.index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        # Compact storage of target records: list of (entity_id, norm_name, norm_addr)
        self.targets: List[Tuple[str, str, str]] = []
        
    def add_target(self, entity_id: str, norm_name: str, norm_addr: str) -> int:
        idx = len(self.targets)
        self.targets.append((entity_id, norm_name, norm_addr))
        keys = generate_blocking_keys(norm_name, norm_addr)
        for k in keys:
            self.index[k].append(idx)
        return idx

    def prune_frequent_keys(self):
        """
        Removes keys with too many entries (high frequency, low selectivity).
        """
        initial_keys = len(self.index)
        pruned_keys = 0
        for k in list(self.index.keys()):
            if len(self.index[k]) > self.max_key_postings:
                del self.index[k]
                pruned_keys += 1
        return initial_keys, pruned_keys

    def get_candidates(self, norm_name: str, norm_addr: str, max_candidates: int = 25) -> List[Tuple[str, str, str]]:
        """
        Retrieves top candidate target records for a query.
        Returns list of (entity_id, norm_name, norm_addr).
        """
        keys = generate_blocking_keys(norm_name, norm_addr)
        candidate_scores: Dict[int, int] = defaultdict(int)
        
        for k in keys:
            postings = self.index.get(k)
            if postings:
                for target_idx in postings:
                    candidate_scores[target_idx] += 1
                    
        if not candidate_scores:
            return []
            
        # Sort by number of matching blocking keys descending
        sorted_indices = sorted(candidate_scores.keys(), key=lambda idx: candidate_scores[idx], reverse=True)
        return [self.targets[idx] for idx in sorted_indices[:max_candidates]]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    
    print("=== Testing Blocking Index ===")
    index = BlockingIndex(max_key_postings=100)
    
    # Add dummy target records
    t1 = index.add_target("S2-001", "zephay labs incorporated", "2621 cotten road tyler tx 75701")
    t2 = index.add_target("S2-002", "orelee s barbershop", "1795 westchester drive high point nc")
    t3 = index.add_target("S2-003", "brahma infosoft private limited", "coimatore colony hunsur karnataka")
    t4 = index.add_target("S3-001", "zephay labs", "2621 cotten rd tyler texas")
    
    print(f"Total targets indexed: {len(index.targets)}")
    candidates = index.get_candidates("zephay labs inc", "2621 cotten road tyler tx")
    print(f"Candidates for 'zephay labs inc': {[c[0] for c in candidates]}")
    assert "S2-001" in [c[0] for c in candidates]
    assert "S3-001" in [c[0] for c in candidates]
    print("[SUCCESS] Blocking module verified.")
