"""One-time script to assign each MIMIC subject a stratified fold (1-10).

Stratification uses MultilabelStratifiedKFold over all diagnostic statements
from report_0..report_17 so that each fold has a balanced distribution of
diagnoses.  An additional column bins ECG-count-per-subject so that heavy
users are spread evenly.

Produces  <data_root>/mimic_folds.csv  with columns: subject_id, strat_fold

Usage:
    python generate_mimic_folds.py --data_root /path/to/mimic_ecg_project \
        [--seed 42] [--n_folds 10] [--min_label_count 20] [--top_k_labels 200]
"""
import argparse
import os

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--record_list_file", type=str, default="record_list.csv")
    parser.add_argument("--machine_measurements_file", type=str,
                        default="machine_measurements.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--min_label_count", type=int, default=20,
                        help="Labels appearing in fewer subjects are dropped")
    parser.add_argument("--top_k_labels", type=int, default=200,
                        help="Keep at most this many labels for stratification")
    parser.add_argument("--ecg_count_bins", type=int, default=5,
                        help="Number of quantile bins for ECG-count-per-subject")
    args = parser.parse_args()

    report_cols = [f"report_{i}" for i in range(18)]

    # ── Load data ──
    records = pd.read_csv(
        os.path.join(args.data_root, args.record_list_file),
        usecols=["subject_id", "study_id"],
    )
    machine = pd.read_csv(
        os.path.join(args.data_root, args.machine_measurements_file),
        usecols=["subject_id", "study_id"] + report_cols,
        low_memory=False,
    )
    df = records.merge(machine, on=["subject_id", "study_id"], how="left")
    print(f"Total ECG records: {len(df)}")

    # ── Collect all labels per subject ──
    # Normalise: lowercase, strip whitespace and trailing periods
    for col in report_cols:
        df[col] = (df[col].fillna("").astype(str).str.strip()
                   .str.lower().str.rstrip(".").str.strip())

    # Melt to long format: (subject_id, label)
    label_long = (
        df.melt(id_vars="subject_id", value_vars=report_cols, value_name="label")
        .drop(columns="variable")
    )
    label_long = label_long[label_long["label"] != ""]
    # Unique labels per subject
    subject_labels = label_long.drop_duplicates(["subject_id", "label"])

    # ── Select labels for stratification ──
    label_counts = subject_labels["label"].value_counts()
    # Filter by minimum count, then keep top-k
    label_counts = label_counts[label_counts >= args.min_label_count]
    selected_labels = label_counts.head(args.top_k_labels).index.tolist()
    print(f"Using {len(selected_labels)} labels for stratification "
          f"(from {label_long['label'].nunique()} total unique)")

    # ── Build subject-level multi-label matrix ──
    subjects = sorted(df["subject_id"].dropna().unique().tolist())
    subject_idx = {s: i for i, s in enumerate(subjects)}
    label_idx = {l: i for i, l in enumerate(selected_labels)}
    n_subjects = len(subjects)
    n_labels = len(selected_labels)

    # +1 column for ECG-count bin
    n_cols = n_labels + args.ecg_count_bins
    Y = np.zeros((n_subjects, n_cols), dtype=np.int8)

    # Fill label columns
    filtered = subject_labels[subject_labels["label"].isin(label_idx)]
    for row in filtered.itertuples(index=False):
        si = subject_idx.get(row.subject_id)
        li = label_idx.get(row.label)
        if si is not None and li is not None:
            Y[si, li] = 1

    # Fill ECG-count bin column (one-hot)
    ecg_counts = df.groupby("subject_id").size()
    ecg_count_arr = np.array([ecg_counts.get(s, 1) for s in subjects])
    bins = pd.qcut(ecg_count_arr, q=args.ecg_count_bins, labels=False, duplicates="drop")
    for i, b in enumerate(bins):
        Y[i, n_labels + b] = 1

    print(f"Label matrix shape: {Y.shape}  "
          f"(avg labels/subject: {Y[:, :n_labels].sum(axis=1).mean():.1f})")

    # ── Multilabel Stratified K-Fold ──
    X = np.arange(n_subjects)
    mskf = MultilabelStratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                     random_state=args.seed)

    fold_assignment = np.zeros(n_subjects, dtype=np.int32)
    for fold_idx, (_, test_idx) in enumerate(mskf.split(X, Y), start=1):
        fold_assignment[test_idx] = fold_idx

    out_df = pd.DataFrame({"subject_id": subjects, "strat_fold": fold_assignment})
    out_df = out_df.sort_values("subject_id").reset_index(drop=True)

    # ── Diagnostics ──
    print(f"\nFold sizes (subjects):")
    print(out_df["strat_fold"].value_counts().sort_index().to_string())

    # Check label balance across folds
    merged = subject_labels.merge(out_df, on="subject_id")
    merged = merged[merged["label"].isin(selected_labels[:10])]  # top 10 only
    pivot = merged.groupby(["strat_fold", "label"]).size().unstack(fill_value=0)
    print(f"\nTop-10 label counts per fold:")
    print(pivot.to_string())

    # ── Save ──
    out_path = os.path.join(args.data_root, "mimic_folds.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} fold assignments to {out_path}")


if __name__ == "__main__":
    main()
