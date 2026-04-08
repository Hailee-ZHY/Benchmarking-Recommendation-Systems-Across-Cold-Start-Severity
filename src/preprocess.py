"""
preprocess.py
-------------
Preprocessing pipeline that plugs into the existing PySpark data loading
from DataProcesser.py / DataLoader.py.

Expected inputs (already produced by teammate's DataLoader.py):
  - my_amazon_books.parquet       (review data: user_id, parent_asin, rating, ...)
  - my_amazon_books_meta.parquet  (metadata:    parent_asin, title, description, ...)

What this module does on top:
  1. Reads those parquet files via PySpark (same session config as DataProcesser.py)
  2. Cleans and k-core filters interactions (min 10 ratings per user/item)
  3. Splits 70/30 train/test (same as teammate's randomSplit)
  4. Builds integer index maps (user_id -> user_idx, parent_asin -> item_idx)
  5. Saves everything to data/processed/ as pandas parquet + pickle

Column naming convention used throughout all other modules:
  user_id, parent_asin, rating   <- original column names kept
  user_idx, item_idx             <- integer indices added here
"""

import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, trim

log = logging.getLogger(__name__)

PROC_DIR = Path("data/processed")

# Cold-start K% levels used throughout the project
COLD_START_LEVELS = [0, 2, 5, 10, 20]

# k-core thresholds
MIN_USER_RATINGS = 10
MIN_ITEM_RATINGS = 10


# ── Spark session (same config as teammate's DataProcesser.py) ─────────────────

def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Amazon_book_recommendation")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.default.parallelism", "64")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.dynamicAllocation.enabled", "false")
        .getOrCreate()
    )


# ── Step 1: Load & clean (mirrors teammate's Data_Process) ─────────────────────

def load_and_clean(spark: SparkSession,
                   review_parquet: str = "my_amazon_books.parquet",
                   meta_parquet:   str = "my_amazon_books_meta.parquet",
                   size: str = "full"):
    """
    Load and clean raw parquet files.

    Parameters
    ----------
    size : "full" | "small"
        "small" limits to 100k rows for fast development (same as teammate).

    Returns
    -------
    df_review : Spark DataFrame  [user_id, parent_asin, rating]
    df_meta   : Spark DataFrame  [parent_asin, title, description]
    """
    log.info(f"Loading review parquet: {review_parquet}")
    df_review_raw = spark.read.parquet(review_parquet)

    # Mirror teammate's cleaning: drop nulls and blank strings
    df_review = (
        df_review_raw
        .filter(
            col("user_id").isNotNull() &
            col("parent_asin").isNotNull() &
            (trim(col("user_id")) != "") &
            (trim(col("parent_asin")) != "")
        )
        .select("user_id", "parent_asin", "rating")
    )

    if size == "small":
        df_review = df_review.limit(100_000)
        log.info("  size='small': limited to 100k rows")

    log.info(f"Loading meta parquet: {meta_parquet}")
    df_meta = spark.read.parquet(meta_parquet).select(
        "parent_asin",
        col("title").cast("string"),
        F.when(col("description").isNotNull(), col("description").cast("string"))
         .otherwise(F.lit("")).alias("description"),
    )

    return df_review, df_meta


# ── Step 2: k-core filtering in Spark ─────────────────────────────────────────

def kcore_filter(df, min_user: int = MIN_USER_RATINGS,
                 min_item: int = MIN_ITEM_RATINGS):
    """
    Iterative k-core filtering done entirely in Spark.
    Extends teammate's one-shot alpha filter to full iterative convergence.
    """
    log.info(f"k-core filtering (min_user={min_user}, min_item={min_item}) ...")
    prev_count = df.count()
    iteration  = 0

    while True:
        iteration += 1
        user_counts = df.groupBy("user_id").agg(F.count("rating").alias("u_cnt"))
        item_counts = df.groupBy("parent_asin").agg(F.count("rating").alias("i_cnt"))

        df = (
            df
            .join(user_counts.filter(col("u_cnt") >= min_user).select("user_id"),
                  on="user_id", how="inner")
            .join(item_counts.filter(col("i_cnt") >= min_item).select("parent_asin"),
                  on="parent_asin", how="inner")
        )

        cur_count = df.count()
        log.info(f"  iter {iteration}: {cur_count:,} rows remaining")
        if cur_count == prev_count:
            break
        prev_count = cur_count

    return df


# ── Step 3: Train / test split (matches teammate's randomSplit) ────────────────

def split_train_test(df, train_ratio: float = 0.7, seed: int = 42):
    """70/30 random split — identical to teammate's randomSplit([0.7, 0.3])."""
    train_df, test_df = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    log.info(f"  Train: {train_df.count():,}  |  Test: {test_df.count():,}")
    return train_df, test_df


# ── Step 4: Collect to pandas + build integer indices ─────────────────────────

def to_pandas_and_index(train_spark, test_spark, meta_spark):
    """
    Collect Spark DataFrames to pandas and add integer user_idx / item_idx.
    Index maps are built from training vocab only.
    Test rows whose user/item is not in training are dropped
    (mirrors teammate's ALS coldStartStrategy='drop').
    """
    log.info("Collecting Spark DataFrames to pandas ...")
    train = train_spark.toPandas()
    test  = test_spark.toPandas()
    meta  = meta_spark.toPandas()

    # Build index maps from training vocab only
    users    = sorted(train["user_id"].unique())
    items    = sorted(train["parent_asin"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}

    def _apply_index(df):
        df = df.copy()
        df["user_idx"] = df["user_id"].map(user2idx)
        df["item_idx"] = df["parent_asin"].map(item2idx)
        df = df.dropna(subset=["user_idx", "item_idx"])
        return df.astype({"user_idx": int, "item_idx": int}).reset_index(drop=True)

    train = _apply_index(train)
    test  = _apply_index(test)

    # Align metadata to training vocab
    meta["item_idx"] = meta["parent_asin"].map(item2idx)
    meta = meta.dropna(subset=["item_idx"]).astype({"item_idx": int})
    meta["text"] = (meta["title"].fillna("") + " " +
                    meta["description"].fillna("")).str.strip()

    log.info(f"  {len(user2idx):,} users, {len(item2idx):,} items")
    log.info(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test, meta, user2idx, item2idx


# ── Step 5: Cold-start simulation ─────────────────────────────────────────────

def simulate_cold_start(test_df: pd.DataFrame,
                        k_pct: int,
                        min_alpha: int = 10,
                        seed: int = 42) -> pd.DataFrame:
    """
    For a given K%, tag each test row as seed or ground-truth.

    Extends teammate's simulate() (which only filtered, didn't split):
      - Only users with >= min_alpha test ratings are included
        (same role as teammate's alpha parameter)
      - is_seed = True  : exposed to the model as input
      - is_gt   = True  : held out as ground truth for evaluation

    K=0 means no seed at all (fully cold user).

    Parameters
    ----------
    test_df   : output of to_pandas_and_index (has user_idx, item_idx)
    k_pct     : 0 | 2 | 5 | 10 | 20
    min_alpha : minimum ratings per user to be included (teammate's alpha=10)
    """
    rng = np.random.default_rng(seed)

    # Filter to eligible users (same as teammate's filtered_user_id logic)
    user_counts = test_df.groupby("user_idx")["rating"].count()
    eligible    = user_counts[user_counts >= min_alpha].index
    df          = test_df[test_df["user_idx"].isin(eligible)].copy()

    df["is_seed"] = False
    df["is_gt"]   = False

    for uid, group in df.groupby("user_idx"):
        n      = len(group)
        n_seed = math.floor(n * k_pct / 100) if k_pct > 0 else 0
        # Random shuffle within user so seed is not position-biased
        shuffled = rng.permutation(group.index.tolist())
        df.loc[shuffled[:n_seed], "is_seed"] = True
        df.loc[shuffled[n_seed:], "is_gt"]   = True

    log.info(f"  K={k_pct}%: {len(eligible):,} eligible users, "
             f"{df['is_seed'].sum():,} seed rows, {df['is_gt'].sum():,} gt rows")
    return df.reset_index(drop=True)


# ── Save / Load helpers ────────────────────────────────────────────────────────

def save_processed(train: pd.DataFrame, test: pd.DataFrame,
                   meta: pd.DataFrame, user2idx: dict, item2idx: dict,
                   proc_dir: Path = PROC_DIR):
    proc_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(proc_dir / "train_interactions.parquet", index=False)
    test.to_parquet(proc_dir  / "test_interactions.parquet",  index=False)
    meta.to_parquet(proc_dir  / "items.parquet",              index=False)
    with open(proc_dir / "user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f)
    with open(proc_dir / "item2idx.pkl", "wb") as f:
        pickle.dump(item2idx, f)
    log.info(f"Saved processed data to {proc_dir}/")


def load_processed(proc_dir: Path = PROC_DIR):
    """Return (train_df, test_df, items_df, user2idx, item2idx)."""
    train = pd.read_parquet(proc_dir / "train_interactions.parquet")
    test  = pd.read_parquet(proc_dir / "test_interactions.parquet")
    items = pd.read_parquet(proc_dir / "items.parquet")
    with open(proc_dir / "user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)
    with open(proc_dir / "item2idx.pkl", "rb") as f:
        item2idx = pickle.load(f)
    return train, test, items, user2idx, item2idx


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_preprocessing(review_parquet: str = "my_amazon_books.parquet",
                      meta_parquet:   str = "my_amazon_books_meta.parquet",
                      size: str = "full",
                      proc_dir: Path = PROC_DIR) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    size : "full" | "small"
        Use "small" for fast development (100k rows, matches teammate).
    """
    spark = get_spark()
    spark.catalog.clearCache()

    # 1. Load & clean
    df_review, df_meta = load_and_clean(spark, review_parquet, meta_parquet, size=size)

    # 2. k-core filter (iterative, done in Spark)
    df_review = kcore_filter(df_review)

    # 3. 70/30 split matching teammate
    train_spark, test_spark = split_train_test(df_review)

    # 4. Collect to pandas + build indices
    train, test, meta, user2idx, item2idx = to_pandas_and_index(
        train_spark, test_spark, df_meta
    )

    # 5. Save
    save_processed(train, test, meta, user2idx, item2idx, proc_dir)

    stats = {
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "n_train": len(train),
        "n_test":  len(test),
    }
    log.info(f"Preprocessing complete: {stats}")
    spark.stop()
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    stats = run_preprocessing(size="small")
    print(stats)
