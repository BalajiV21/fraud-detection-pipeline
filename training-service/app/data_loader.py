"""Load and join the IEEE-CIS Fraud Detection CSVs."""
import logging
import pandas as pd

from .config import RAW_DIR, TRANSACTION_FILE, IDENTITY_FILE, TIME_COL, SAMPLE_FRAC, RANDOM_STATE

log = logging.getLogger(__name__)


def load_raw() -> pd.DataFrame:
    """Load both tables, left-join on TransactionID, return combined frame."""
    txn_path = RAW_DIR / TRANSACTION_FILE
    id_path = RAW_DIR / IDENTITY_FILE

    if not txn_path.exists():
        raise FileNotFoundError(
            f"Missing {txn_path}. Place the Kaggle IEEE-CIS files in data/raw/."
        )
    if not id_path.exists():
        raise FileNotFoundError(
            f"Missing {id_path}. Place the Kaggle IEEE-CIS files in data/raw/."
        )

    log.info("Loading %s", txn_path)
    #txn = pd.read_csv(txn_path)
    txn = pd.read_csv(txn_path, nrows=180000)
    log.info("Loading %s", id_path)
    ident = pd.read_csv(id_path)

    log.info("txn shape=%s, identity shape=%s", txn.shape, ident.shape)
    df = txn.merge(ident, on="TransactionID", how="left")
    log.info("Joined shape=%s", df.shape)

    # Sort by time so downstream time-based split works
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if SAMPLE_FRAC < 1.0:
        log.warning("Sampling %.2f of data for fast dev", SAMPLE_FRAC)
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).sort_values(TIME_COL).reset_index(drop=True)

    return df
