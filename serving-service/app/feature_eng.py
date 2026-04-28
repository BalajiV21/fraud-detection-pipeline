"""Feature engineering pipeline.

Produces a (features_df, fitted_pipeline) pair. The fitted pipeline is saved
to the shared volume so the serving-service can apply identical transforms.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FeaturePipeline:
    """Stateful feature transformer. Fit on train, transform on val/test/serve."""

    numeric_medians: Dict[str, float] = field(default_factory=dict)
    cat_freq_maps: Dict[str, Dict] = field(default_factory=dict)
    cat_target_maps: Dict[str, Dict] = field(default_factory=dict)
    onehot_columns: List[str] = field(default_factory=list)
    final_columns: List[str] = field(default_factory=list)
    drop_cols: List[str] = field(default_factory=list)
    fitted: bool = False

    LOW_CARD = ["ProductCD", "card4", "card6", "DeviceType"]
    MED_CARD = ["P_emaildomain", "R_emaildomain", "card1", "card2", "card3", "card5"]
    HIGH_NULL_THRESHOLD = 0.90
    NULL_FLAG_RANGE = (0.10, 0.50)

    # ---------- public ----------
    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        out = self._engineer_basic(df.copy())
        out = self._fit_dropcols(out)
        out = self._fit_null_flags(out)
        out = self._fit_numeric(out)
        out = self._fit_categorical(out, target)
        out = self._finalize(out, fit=True)
        self.fitted = True
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted")
        out = self._engineer_basic(df.copy())
        out = out.drop(columns=[c for c in self.drop_cols if c in out.columns], errors="ignore")
        out = self._apply_null_flags(out)
        out = self._apply_numeric(out)
        out = self._apply_categorical(out)
        out = self._finalize(out, fit=False)
        return out

    # ---------- internal ----------
    def _engineer_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        if "TransactionDT" in df.columns:
            df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
            df["day_of_week"] = (df["TransactionDT"] // (3600 * 24)) % 7
        if "TransactionAmt" in df.columns:
            df["amount_log"] = np.log1p(df["TransactionAmt"])
            df["amount_decimal"] = (df["TransactionAmt"] - df["TransactionAmt"].astype(int)).round(4)
        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)
        df["null_count"] = df.isnull().sum(axis=1)
        return df

    def _fit_dropcols(self, df: pd.DataFrame) -> pd.DataFrame:
        null_frac = df.isnull().mean()
        self.drop_cols = null_frac[null_frac > self.HIGH_NULL_THRESHOLD].index.tolist()
        log.info("Dropping %d high-null columns (>%.0f%% missing)", len(self.drop_cols), self.HIGH_NULL_THRESHOLD * 100)
        return df.drop(columns=self.drop_cols, errors="ignore")

    def _fit_null_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        null_frac = df.isnull().mean()
        flag_cols = null_frac[(null_frac >= self.NULL_FLAG_RANGE[0]) & (null_frac <= self.NULL_FLAG_RANGE[1])].index
        self._null_flag_cols = list(flag_cols)
        for c in self._null_flag_cols:
            df[f"{c}__isnull"] = df[c].isnull().astype(int)
        return df

    def _apply_null_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in getattr(self, "_null_flag_cols", []):
            if c in df.columns:
                df[f"{c}__isnull"] = df[c].isnull().astype(int)
            else:
                df[f"{c}__isnull"] = 1
        return df

    def _fit_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number]).columns
        for c in numeric:
            self.numeric_medians[c] = float(df[c].median())
        return self._apply_numeric(df)

    def _apply_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        for c, m in self.numeric_medians.items():
            if c in df.columns:
                df[c] = df[c].fillna(m)
        return df

    def _fit_categorical(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        # One-hot for low-card
        for col in self.LOW_CARD:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna("Unknown").astype(str)
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False).astype(int)
            self.onehot_columns.extend(dummies.columns.tolist())
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        # Target encoding for medium-card (simple, no k-fold to keep scaffold lean — see TODO)
        # NOTE: This is a simplification. For interview-grade rigor, replace with k-fold target encoding.
        for col in self.MED_CARD:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna("Unknown").astype(str)
            tmap = target.groupby(df[col]).mean().to_dict()
            self.cat_target_maps[col] = tmap
            df[f"{col}__te"] = df[col].map(tmap).fillna(target.mean())
            # Frequency too
            fmap = df[col].value_counts(normalize=True).to_dict()
            self.cat_freq_maps[col] = fmap
            df[f"{col}__freq"] = df[col].map(fmap).fillna(0.0)
            df = df.drop(columns=[col])

        # Drop any remaining non-numeric columns (DeviceInfo, id_30, id_31, etc — just frequency-encode them)
        remaining_obj = df.select_dtypes(include=["object"]).columns.tolist()
        for col in remaining_obj:
            df[col] = df[col].fillna("Unknown").astype(str)
            fmap = df[col].value_counts(normalize=True).to_dict()
            self.cat_freq_maps[col] = fmap
            df[f"{col}__freq"] = df[col].map(fmap).fillna(0.0)
            df = df.drop(columns=[col])

        return df

    def _apply_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        # One-hot
        for col in self.LOW_CARD:
            if col not in df.columns:
                # rebuild empty dummies
                continue
            df[col] = df[col].fillna("Unknown").astype(str)
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        # Target encoding
        for col, tmap in self.cat_target_maps.items():
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)
                global_mean = float(np.mean(list(tmap.values()))) if tmap else 0.0
                df[f"{col}__te"] = df[col].map(tmap).fillna(global_mean)
                df = df.drop(columns=[col])
            else:
                df[f"{col}__te"] = 0.0

        # Frequency encoding (medium + remaining)
        for col, fmap in self.cat_freq_maps.items():
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)
                df[f"{col}__freq"] = df[col].map(fmap).fillna(0.0)
                df = df.drop(columns=[col])
            else:
                if f"{col}__freq" not in df.columns:
                    df[f"{col}__freq"] = 0.0

        return df

    def _finalize(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        # Drop ID columns from features
        for c in ["TransactionID", "isFraud"]:
            if c in df.columns:
                df = df.drop(columns=[c])
        # Coerce all to numeric
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if fit:
            self.final_columns = df.columns.tolist()
        else:
            # Align columns to training schema
            for c in self.final_columns:
                if c not in df.columns:
                    df[c] = 0.0
            df = df[self.final_columns]
        return df
