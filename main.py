"""
main.py
-------
Full pipeline integrating teammate's DataLoader.py / DataProcesser.py / ALS.py
with our Semantic, DropoutNet, cold-start simulation, and evaluation.

Steps
-----
  1. preprocess  — PySpark: load parquets → k-core filter → 70/30 split → save
  2. embeddings  — Sentence-BERT item embeddings (written to memmap)
  3. train       — ALS (PySpark, from ALS.py) + Semantic + DropoutNet
  4. evaluate    — cold-start simulation K ∈ {0,2,5,10,20}, Recall@10 / NDCG@10
  5. analyze     — degradation curves, crossover detection

Quick-start
-----------
  # Fast dev run (100k rows, 500 test users):
  python main.py --size small --max-users 500

  # Full run:
  python main.py

  # Skip already-done steps:
  python main.py --skip-preprocess --skip-embeddings --skip-train
"""

import argparse
import logging
from pathlib import Path

from src.preprocess       import run_preprocessing, load_processed, COLD_START_LEVELS
from src.embeddings       import build_item_embeddings, load_item_embeddings
from src.model_als        import ALSModel
from src.model_semantic   import SemanticModel
from src.model_dropoutnet import DropoutNetModel, extract_als_factors
from src.evaluate         import run_full_evaluation
from src.analyze          import run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROC_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")

REVIEW_PARQUET = "my_amazon_books.parquet"
META_PARQUET   = "my_amazon_books_meta.parquet"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Cold-start benchmarking pipeline")
    p.add_argument("--size", choices=["full", "small"], default="full",
                   help="'small' limits to 100k rows (matches teammate's dev mode)")
    p.add_argument("--skip-preprocess",  action="store_true")
    p.add_argument("--skip-embeddings",  action="store_true")
    p.add_argument("--skip-train",       action="store_true")
    p.add_argument("--skip-evaluate",    action="store_true")
    p.add_argument("--skip-analyze",     action="store_true")
    p.add_argument("--top-k",   type=int, default=10)
    p.add_argument("--max-users", type=int, default=None,
                   help="Cap test users per K level (speeds up evaluation)")
    return p.parse_args()


# ── Step 1: Preprocessing ──────────────────────────────────────────────────────

def step_preprocess(size: str):
    stats = run_preprocessing(
        review_parquet=REVIEW_PARQUET,
        meta_parquet=META_PARQUET,
        size=size,
        proc_dir=PROC_DIR,
    )
    log.info(f"Preprocessing stats: {stats}")


# ── Step 2: Embeddings ─────────────────────────────────────────────────────────

def step_embeddings():
    build_item_embeddings(proc_dir=PROC_DIR, force=False)


# ── Step 3: Training ───────────────────────────────────────────────────────────

def step_train(size: str):
    """
    Train all three models.

    ALS is trained via the `implicit` library so we have direct numpy access
    to the factor matrices needed by DropoutNet.
    Semantic just loads SBERT embeddings + computes item popularity.
    DropoutNet trains the two-tower MLP using ALS factors + SBERT embeddings.
    """
    train_df, test_df, items_df, user2idx, item2idx = load_processed(PROC_DIR)
    n_users = len(user2idx)
    n_items = len(item2idx)

    # ── ALS ────────────────────────────────────────────────────────────────────
    # We use our own ALSModel wrapper (which uses the `implicit` library)
    # so that we have numpy access to the factor matrices for DropoutNet.
    # Teammate's PySpark ALS is great for scale but doesn't expose numpy factors
    # easily at inference time.  Both produce equivalent results.
    als_path = PROC_DIR / "als_model.pkl"
    if als_path.exists():
        log.info("ALS model found — loading from disk.")
        als_model = ALSModel.load(als_path)
    else:
        als_model = ALSModel(
            factors=64 if size == "small" else 128,
            iterations=5 if size == "small" else 20,
        )
        als_model.train(train_df, n_users, n_items)
        als_model.save(als_path)

    # ── Semantic ────────────────────────────────────────────────────────────────
    sem_path = PROC_DIR / "semantic_model.pkl"
    if sem_path.exists():
        log.info("Semantic model found — loading from disk.")
        sem_model = SemanticModel.load(sem_path)
    else:
        sem_model = SemanticModel()
        sem_model.train(train_df, n_items, proc_dir=PROC_DIR)
        sem_model.save(sem_path)

    # ── DropoutNet ──────────────────────────────────────────────────────────────
    dnet_path = PROC_DIR / "dropoutnet_model.pt"
    if dnet_path.exists():
        log.info("DropoutNet found — loading from disk.")
        dnet_model = DropoutNetModel.load(dnet_path)
    else:
        item_emb     = load_item_embeddings(PROC_DIR)
        user_factors = als_model.model.user_factors   # (n_users, F) numpy
        item_factors = als_model.item_factors          # (n_items, F) numpy

        dnet_model = DropoutNetModel(
            epochs=5 if size == "small" else 20,
            batch_size=512 if size == "small" else 1024,
        )
        dnet_model.train(train_df, user_factors, item_factors,
                         item_emb, proc_dir=PROC_DIR)
        dnet_model.save(dnet_path)

    return als_model, sem_model, dnet_model


# ── Step 4: Evaluation ─────────────────────────────────────────────────────────

def step_evaluate(als_model, sem_model, dnet_model, top_k: int,
                  max_users: int | None):
    _, test_df, _, _, _ = load_processed(PROC_DIR)

    return run_full_evaluation(
        als_model        = als_model,
        semantic_model   = sem_model,
        dropoutnet_model = dnet_model,
        test_df          = test_df,
        cold_start_levels= COLD_START_LEVELS,
        top_k            = top_k,
        max_users        = max_users,
        results_dir      = RESULTS_DIR,
    )


# ── Step 5: Analysis ───────────────────────────────────────────────────────────

def step_analyze(top_k: int):
    run_analysis(results_dir=RESULTS_DIR, top_k=top_k)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.skip_preprocess:
        log.info("━━━ STEP 1: Preprocessing (PySpark) ━━━")
        step_preprocess(size=args.size)
    else:
        log.info("Skipping preprocessing.")

    if not args.skip_embeddings:
        log.info("━━━ STEP 2: Building SBERT item embeddings ━━━")
        step_embeddings()
    else:
        log.info("Skipping embeddings.")

    if not args.skip_train:
        log.info("━━━ STEP 3: Training models ━━━")
        als_model, sem_model, dnet_model = step_train(size=args.size)
    else:
        log.info("Loading saved models ...")
        als_model  = ALSModel.load(PROC_DIR / "als_model.pkl")
        sem_model  = SemanticModel.load(PROC_DIR / "semantic_model.pkl")
        dnet_model = DropoutNetModel.load(PROC_DIR / "dropoutnet_model.pt")

    if not args.skip_evaluate:
        log.info("━━━ STEP 4: Cold-start evaluation ━━━")
        step_evaluate(als_model, sem_model, dnet_model,
                      top_k=args.top_k, max_users=args.max_users)
    else:
        log.info("Skipping evaluation.")

    if not args.skip_analyze:
        log.info("━━━ STEP 5: Analysis & plots ━━━")
        step_analyze(top_k=args.top_k)
    else:
        log.info("Skipping analysis.")

    log.info("Pipeline complete. Check results/")


if __name__ == "__main__":
    main()