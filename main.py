import csv
import os
import random
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "intfloat/multilingual-e5-small"
RANDOM_SEED = 42

# The code never loads the 10M-row target set into pandas.
# SQLite keeps the target data on disk and returns only small
# candidate sets.
DB_DIR = "work"
DB_FILE = "entity_resolution.sqlite"

TARGET_CHUNK = 5000
TRAIN_SAMPLE = 6000
VALIDATION_SAMPLE = 1500

MAX_BLOCK_CANDIDATES = 250
CHEAP_TOP_K = 20
FINAL_TOP_K = 5

# For training/validation, include all true matches even if
# blocking did not retrieve them.
MAX_TRUE_MATCHES_PER_QUERY = 20

EMBED_BATCH = 32

# Search a few thresholds for F0.5.
THRESHOLDS = np.arange(0.30, 0.951, 0.02)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"
WORK_DIR = BASE_DIR / DB_DIR
DB_PATH = WORK_DIR / DB_FILE
PHASE_FILE = WORK_DIR / "phase.txt"

# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_text(value):
    if value is None:
        return ""
    value = str(value).strip()
    value = unicodedata.normalize("NFKC", value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def first_token(text):
    parts = text.split()
    return parts[0] if parts else ""


def key_prefix(text, n):
    compact = text.replace(" ", "")
    return compact[:n]


def postal_codes(text):
    if not text:
        return []
    return list(dict.fromkeys(
        re.findall(r"\b[a-z]?\d{4,6}[a-z]?\b", text.lower())
    ))


def record_from_dict(row):
    # Strip BOM from a possible first header value.
    clean = {}
    for k, v in row.items():
        if k is None:
            continue
        clean[k.strip().lstrip("\ufeff")] = v if v is not None else ""

    required = ["entity_id", "business_name", "business_address", "country"]
    missing = [x for x in required if x not in clean]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Available columns: {list(clean.keys())}"
        )

    entity_id = str(clean["entity_id"]).strip()
    name = normalize_text(clean["business_name"])
    address = normalize_text(clean["business_address"])
    country = normalize_text(clean["country"])
    postal = postal_codes(address)

    return {
        "entity_id": entity_id,
        "name": name,
        "address": address,
        "country": country,
        "postal": postal[0] if postal else "",
        "name4": key_prefix(name, 4),
        "name6": key_prefix(name, 6),
        "addr6": key_prefix(address, 6),
        "name_first": first_token(name),
        "addr_first": first_token(address),
    }


def read_tsv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield record_from_dict(row)

# ============================================================
# SQLITE DATABASE
# ============================================================
def connect_db():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("PRAGMA cache_size=-65536")  # ~64 MB cache
    return con


def create_schema(con):
    con.execute("DROP TABLE IF EXISTS target")
    con.execute("DROP TABLE IF EXISTS truth")

    con.execute("""
        CREATE TABLE target (
            rid INTEGER PRIMARY KEY,
            entity_id TEXT NOT NULL,
            name TEXT,
            address TEXT,
            country TEXT,
            postal TEXT,
            name4 TEXT,
            name6 TEXT,
            addr6 TEXT,
            name_first TEXT,
            addr_first TEXT
        )
    """)

    con.execute("""
        CREATE TABLE truth (
            source1_id TEXT PRIMARY KEY,
            matched_ids TEXT
        )
    """)
    con.commit()


def build_target_database(source_paths, build_truth_path=None):
    print("\nBuilding disk-backed target index...")
    print(f"Database: {DB_PATH}")

    con = connect_db()
    create_schema(con)

    insert_sql = """
        INSERT INTO target
        (entity_id,name,address,country,postal,name4,name6,addr6,name_first,addr_first)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """

    total = 0
    t0 = time.time()

    for path in source_paths:
        print(f"Indexing target file: {path}")
        batch = []
        for rec in read_tsv(path):
            batch.append((
                rec["entity_id"], rec["name"], rec["address"], rec["country"],
                rec["postal"], rec["name4"], rec["name6"], rec["addr6"],
                rec["name_first"], rec["addr_first"]
            ))
            if len(batch) >= TARGET_CHUNK:
                con.executemany(insert_sql, batch)
                total += len(batch)
                batch.clear()
                if total % 500000 < TARGET_CHUNK:
                    print(f"  indexed {total:,} rows")
        if batch:
            con.executemany(insert_sql, batch)
            total += len(batch)
        con.commit()

    print(f"Target rows indexed: {total:,}")
    print("Creating SQLite indexes...")

    for col in ["postal", "name6", "name4", "addr6", "name_first", "addr_first"]:
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_target_{col} ON target({col})")
    con.execute("CREATE INDEX IF NOT EXISTS idx_target_entity ON target(entity_id)")
    con.commit()

    if build_truth_path is not None:
        print(f"Indexing ground truth: {build_truth_path}")
        truth_batch = []
        with open(build_truth_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fields = [x.strip().lstrip("\ufeff") for x in (reader.fieldnames or [])]
            if "source1_entity_id" not in fields or "matched_entity_ids" not in fields:
                raise ValueError(
                    f"Ground truth must contain source1_entity_id and matched_entity_ids. Found {fields}"
                )
            for row in reader:
                row = {k.strip().lstrip("\ufeff"): v for k, v in row.items() if k is not None}
                sid = str(row.get("source1_entity_id", "")).strip()
                mids = str(row.get("matched_entity_ids", "")).strip()
                truth_batch.append((sid, mids))
                if len(truth_batch) >= TARGET_CHUNK:
                    con.executemany("INSERT OR REPLACE INTO truth VALUES (?,?)", truth_batch)
                    truth_batch.clear()
            if truth_batch:
                con.executemany("INSERT OR REPLACE INTO truth VALUES (?,?)", truth_batch)
        con.commit()
        print("Ground truth indexed.")

    con.close()
    print(f"Index build time: {(time.time()-t0)/60:.1f} minutes")


def ensure_train_db():
    # A previous completed run leaves the SQLite file containing test targets.
    # Rebuild the training index unless the database is explicitly marked
    # as being in the training phase.
    if PHASE_FILE.exists() and PHASE_FILE.read_text(encoding="utf-8").strip() == "train" and DB_PATH.exists():
        print("Existing training index found; reusing it.")
        return

    for p in [DB_PATH, Path(str(DB_PATH) + "-wal"), Path(str(DB_PATH) + "-shm")]:
        if p.exists():
            try:
                p.unlink()
            except PermissionError:
                pass

    build_target_database(
        [TRAIN_DIR / "train_source2.tsv", TRAIN_DIR / "train_source3.tsv"],
        TRAIN_DIR / "train_ground_truth.tsv"
    )
    PHASE_FILE.write_text("train", encoding="utf-8")


def switch_to_test_db():
    # Reuse the same database file, so only one ~10M-row target index
    # exists on disk at a time.
    con = connect_db()
    con.execute("DROP TABLE IF EXISTS target")
    con.execute("DROP TABLE IF EXISTS truth")
    con.commit()
    con.close()

    build_target_database(
        [TEST_DIR / "test_source2.tsv", TEST_DIR / "test_source3.tsv"],
        None
    )
    PHASE_FILE.write_text("test", encoding="utf-8")

# ============================================================
# GROUND TRUTH
# ============================================================
def parse_ids(text):
    text = str(text or "").strip()
    if not text:
        return []
    # Accept comma/semicolon/pipe separated variants.
    parts = re.split(r"[,;|]", text)
    return [x.strip() for x in parts if x.strip()]


def get_truth(con, source1_id):
    row = con.execute(
        "SELECT matched_ids FROM truth WHERE source1_id=?",
        (source1_id,)
    ).fetchone()
    if not row:
        return set()
    return set(parse_ids(row[0]))

# ============================================================
# SOURCE-1 SAMPLING
# ============================================================
def reservoir_sample(path, n, seed=42):
    rng = random.Random(seed)
    sample = []
    seen = 0
    for rec in read_tsv(path):
        seen += 1
        if len(sample) < n:
            sample.append(rec)
        else:
            j = rng.randrange(seen)
            if j < n:
                sample[j] = rec
    return sample

# ============================================================
# CANDIDATE GENERATION
# ============================================================
def fetch_rows(con, sql, params, limit):
    sql = sql + " LIMIT ?"
    rows = con.execute(sql, tuple(params) + (limit,)).fetchall()
    return rows


def candidate_rows(con, q, max_candidates=MAX_BLOCK_CANDIDATES):
    # Return target rows. Each tuple is:
    # rid, entity_id, name, address, country, postal, ...
    found = {}

    blocks = [
        ("postal", q["postal"], 60),
        ("name6", q["name6"], 80),
        ("name4", q["name4"], 80),
        ("addr6", q["addr6"], 60),
        ("name_first", q["name_first"], 80),
        ("addr_first", q["addr_first"], 50),
    ]

    for col, value, limit in blocks:
        if not value:
            continue
        rows = fetch_rows(
            con,
            f"SELECT rid,entity_id,name,address,country,postal,name4,name6,addr6,name_first,addr_first FROM target WHERE {col}=?",
            (value,),
            limit
        )
        for row in rows:
            found[row[0]] = row
            if len(found) >= max_candidates:
                break
        if len(found) >= max_candidates:
            break

    return list(found.values())


def cheap_score(q, t):
    name = fuzz.token_set_ratio(q["name"], t[2]) / 100.0 if q["name"] and t[2] else 0.0
    name_ratio = fuzz.ratio(q["name"], t[2]) / 100.0 if q["name"] and t[2] else 0.0
    addr = fuzz.token_set_ratio(q["address"], t[3]) / 100.0 if q["address"] and t[3] else 0.0
    addr_ratio = fuzz.ratio(q["address"], t[3]) / 100.0 if q["address"] and t[3] else 0.0
    country = 1.0 if q["country"] and q["country"] == t[4] else 0.0
    postal = 1.0 if q["postal"] and q["postal"] == t[5] else 0.0
    return (
        0.45 * name
        + 0.20 * name_ratio
        + 0.20 * addr
        + 0.05 * addr_ratio
        + 0.07 * country
        + 0.03 * postal
    )


def top_candidates(con, q, k=CHEAP_TOP_K):
    rows = candidate_rows(con, q)
    scored = [(cheap_score(q, r), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]

# ============================================================
# E5 MODEL
# ============================================================
class E5Encoder:
    def __init__(self):
        print("\nLoading pretrained model:")
        print(MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME)

    @staticmethod
    def query_text(r):
        return (
            "query: name " + r["name"]
            + " address " + r["address"]
            + " country " + r["country"]
        )

    @staticmethod
    def target_text(row):
        # SQLite target row format.
        return (
            "passage: name " + (row[2] or "")
            + " address " + (row[3] or "")
            + " country " + (row[4] or "")
        )

    def encode(self, texts):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)

# ============================================================
# FEATURES
# ============================================================
def make_pair_features(q, t, semantic):
    name_ratio = fuzz.ratio(q["name"], t[2]) / 100.0 if q["name"] and t[2] else 0.0
    name_token = fuzz.token_set_ratio(q["name"], t[2]) / 100.0 if q["name"] and t[2] else 0.0
    name_sort = fuzz.token_sort_ratio(q["name"], t[2]) / 100.0 if q["name"] and t[2] else 0.0

    addr_ratio = fuzz.ratio(q["address"], t[3]) / 100.0 if q["address"] and t[3] else 0.0
    addr_token = fuzz.token_set_ratio(q["address"], t[3]) / 100.0 if q["address"] and t[3] else 0.0

    country = float(bool(q["country"]) and q["country"] == t[4])
    postal = float(bool(q["postal"]) and q["postal"] == t[5])

    name_len = min(len(q["name"]), len(t[2])) / max(1, max(len(q["name"]), len(t[2])))
    addr_len = min(len(q["address"]), len(t[3])) / max(1, max(len(q["address"]), len(t[3])))

    return np.array([
        name_ratio,
        name_token,
        name_sort,
        addr_ratio,
        addr_token,
        country,
        postal,
        semantic,
        name_len,
        addr_len,
    ], dtype=np.float32)

# ============================================================
# PAIR DATASET FOR TRAINING / VALIDATION
# ============================================================
def build_pairs(con, queries, encoder, include_truth=True):
    pair_rows = []
    labels = []

    for qi, q in enumerate(queries):
        candidates = top_candidates(con, q, CHEAP_TOP_K)

        if include_truth:
            true_ids = get_truth(con, q["entity_id"])
            # Add true target records directly by entity_id so a
            # blocking miss does not become a false negative.
            for tid in list(true_ids)[:MAX_TRUE_MATCHES_PER_QUERY]:
                row = con.execute(
                    "SELECT rid,entity_id,name,address,country,postal,name4,name6,addr6,name_first,addr_first FROM target WHERE entity_id=? LIMIT 1",
                    (tid,)
                ).fetchone()
                if row is not None:
                    candidates.append(row)

        # Deduplicate by target rid.
        uniq = {}
        for row in candidates:
            uniq[row[0]] = row
        candidates = list(uniq.values())

        if not candidates:
            continue

        # Keep training candidate count small. True matches were
        # inserted above and are retained.
        if len(candidates) > 30:
            true_ids = get_truth(con, q["entity_id"]) if include_truth else set()
            candidates.sort(
                key=lambda r: (r[1] in true_ids, cheap_score(q, r)),
                reverse=True
            )
            candidates = candidates[:30]

        q_text = E5Encoder.query_text(q)
        t_texts = [E5Encoder.target_text(r) for r in candidates]
        q_emb = encoder.encode([q_text])[0]
        t_emb = encoder.encode(t_texts)
        semantic_scores = t_emb @ q_emb

        true_ids = get_truth(con, q["entity_id"]) if include_truth else set()

        for row, sem in zip(candidates, semantic_scores):
            pair_rows.append(make_pair_features(q, row, float(sem)))
            labels.append(1 if row[1] in true_ids else 0)

    if not pair_rows:
        raise RuntimeError("No training/validation candidate pairs were generated.")

    X = np.vstack(pair_rows)
    y = np.asarray(labels, dtype=np.int8)
    return X, y

# ============================================================
# F0.5
# ============================================================
def f05(predicted, actual):
    predicted = set(predicted)
    actual = set(actual)
    if not predicted and not actual:
        return 1.0
    if not predicted:
        return 0.0
    tp = len(predicted & actual)
    precision = tp / len(predicted)
    recall = tp / len(actual) if actual else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def validate_threshold(con, queries, encoder, classifier):
    print("\nGenerating validation predictions...")
    per_query = []

    for i, q in enumerate(queries, 1):
        candidates = top_candidates(con, q, FINAL_TOP_K)
        if not candidates:
            per_query.append((q, [], np.array([], dtype=np.float32)))
            continue

        q_emb = encoder.encode([E5Encoder.query_text(q)])[0]
        t_emb = encoder.encode([E5Encoder.target_text(r) for r in candidates])
        sem = t_emb @ q_emb
        X = np.vstack([
            make_pair_features(q, r, float(s))
            for r, s in zip(candidates, sem)
        ])
        probs = classifier.predict_proba(X)[:, 1]
        per_query.append((q, candidates, probs))

        if i % 500 == 0:
            print(f"  validated {i:,}/{len(queries):,}")

    best_threshold = 0.5
    best_score = -1.0

    for threshold in THRESHOLDS:
        scores = []
        for q, candidates, probs in per_query:
            predicted = [
                r[1] for r, p in zip(candidates, probs)
                if p >= threshold
            ]
            actual = get_truth(con, q["entity_id"])
            scores.append(f05(predicted, actual))
        score = float(np.mean(scores)) if scores else 0.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    print(f"\nValidation macro F0.5: {best_score:.4f}")
    print(f"Selected threshold: {best_threshold:.2f}")
    return best_threshold

# ============================================================
# TRAIN MODEL
# ============================================================
def train_model(con, encoder):
    print("\nSampling training Source 1...")
    sample = reservoir_sample(TRAIN_DIR / "train_source1.tsv", TRAIN_SAMPLE, RANDOM_SEED)
    random.Random(RANDOM_SEED).shuffle(sample)

    split = int(len(sample) * 0.80)
    train_queries = sample[:split]
    val_queries = sample[split:split + VALIDATION_SAMPLE]

    print(f"Training queries: {len(train_queries):,}")
    print(f"Validation queries: {len(val_queries):,}")

    print("\nBuilding training pairs...")
    X_train, y_train = build_pairs(con, train_queries, encoder, include_truth=True)

    print(f"Training pairs: {len(y_train):,}")
    print(f"Positive pairs: {int(y_train.sum()):,}")
    print(f"Negative pairs: {int((y_train == 0).sum()):,}")

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training data contains only one class. Check train_ground_truth.tsv.")

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    classifier.fit(X_train, y_train)

    threshold = validate_threshold(
        con,
        val_queries,
        encoder,
        classifier
    )

    return classifier, threshold

# ============================================================
# TEST PROCESSING
# ============================================================
def write_test_predictions(con, encoder, classifier, threshold):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    # Reservoir is not used here: test must be processed completely.
    total = 0
    matched_count = 0

    with open(TEST_DIR / "test_source1.tsv", "r", encoding="utf-8-sig", newline="") as f, \
         open(matching_path, "w", encoding="utf-8", newline="") as mf, \
         open(candidate_path, "w", encoding="utf-8", newline="") as cf:

        reader = csv.DictReader(f, delimiter="\t")
        mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")

        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])

        batch = []

        def flush(rows):
            nonlocal total, matched_count
            if not rows:
                return

            # First retrieve cheap candidates for each query.
            work = []
            for q in rows:
                candidates = top_candidates(con, q, FINAL_TOP_K)
                work.append((q, candidates))

            # Encode all queries and targets in a compact batch.
            query_texts = [E5Encoder.query_text(q) for q, c in work if c]
            q_embeddings = encoder.encode(query_texts) if query_texts else np.empty((0, 384), dtype=np.float32)

            cursor = 0
            for q, candidates in work:
                if not candidates:
                    mw.writerow([q["entity_id"], ""])
                    cw.writerow([q["entity_id"], ""])
                    total += 1
                    continue

                q_emb = q_embeddings[cursor]
                cursor += 1

                target_embeddings = encoder.encode([
                    E5Encoder.target_text(r) for r in candidates
                ])
                semantic = target_embeddings @ q_emb

                X = np.vstack([
                    make_pair_features(q, r, float(s))
                    for r, s in zip(candidates, semantic)
                ])
                probs = classifier.predict_proba(X)[:, 1]

                # Candidate pairs are exactly the candidates passed to
                # the final classifier.
                candidate_ids = []
                matched_ids = []

                for r, p in zip(candidates, probs):
                    if r[1] not in candidate_ids:
                        candidate_ids.append(r[1])
                    if p >= threshold and r[1] not in matched_ids:
                        matched_ids.append(r[1])

                # Deterministic exact agreement safeguard. This prevents
                # obvious exact name+address pairs from being lost due to
                # calibration of the classifier.
                for r in candidates:
                    exact_name = q["name"] and q["name"] == r[2]
                    exact_address = q["address"] and q["address"] == r[3]
                    same_country = q["country"] and q["country"] == r[4]
                    same_postal = q["postal"] and q["postal"] == r[5]
                    if exact_name and (exact_address or same_postal) and same_country:
                        if r[1] not in matched_ids:
                            matched_ids.append(r[1])

                if matched_ids:
                    matched_count += 1

                mw.writerow([
                    q["entity_id"],
                    ",".join(matched_ids)
                ])
                cw.writerow([
                    q["entity_id"],
                    ",".join(candidate_ids)
                ])
                total += 1

        for row in reader:
            batch.append(record_from_dict(row))
            if len(batch) >= 250:
                flush(batch)
                batch.clear()
                if total % 5000 < 250:
                    print(f"Processed test Source 1: {total:,}")

        flush(batch)

    print("\nTest processing complete.")
    print(f"Test Source 1 rows: {total:,}")
    print(f"Rows with at least one match: {matched_count:,}")
    print(f"Matching output: {matching_path}")
    print(f"Candidate output: {candidate_path}")

# ============================================================
# MAIN
# ============================================================
def check_files():
    required = [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
        TRAIN_DIR / "train_ground_truth.tsv",
        TEST_DIR / "test_source1.tsv",
        TEST_DIR / "test_source2.tsv",
        TEST_DIR / "test_source3.tsv",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required dataset file not found:\n{p}")


def main():
    print("=" * 60)
    print("MEMORY-EFFICIENT BUSINESS ENTITY RESOLUTION")
    print("=" * 60)
    print(f"Project:  {BASE_DIR}")
    print(f"Dataset:  {DATASET_DIR}")
    print(f"Output:   {OUTPUT_DIR}")
    print(f"DB:       {DB_PATH}")

    check_files()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # TRAINING INDEX
    # --------------------------------------------------------
    ensure_train_db()
    con = connect_db()

    # --------------------------------------------------------
    # PRETRAINED MODEL
    # --------------------------------------------------------
    encoder = E5Encoder()

    # --------------------------------------------------------
    # TRAIN CLASSIFIER + SELECT F0.5 THRESHOLD
    # --------------------------------------------------------
    classifier, threshold = train_model(con, encoder)

    con.close()

    # --------------------------------------------------------
    # REBUILD TARGET INDEX USING TEST SOURCE2 + SOURCE3 ONLY
    # --------------------------------------------------------
    print("\nSwitching disk-backed index from training targets to test targets...")
    switch_to_test_db()

    con = connect_db()

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------
    write_test_predictions(
        con,
        encoder,
        classifier,
        threshold
    )

    con.close()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print("Created:")
    print(OUTPUT_DIR / "matching_results.tsv")
    print(OUTPUT_DIR / "candidate_pairs.tsv")


if __name__ == "__main__":
    main()
