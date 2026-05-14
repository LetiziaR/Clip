"""Assign PTB-XL-style superclass labels to MIMIC-IV-ECG subjects.

Superclasses (multi-label per subject):
    NORM  Normal ECG
    MI    Myocardial Infarction
    STTC  ST/T Change
    CD    Conduction Disturbance
    HYP   Hypertrophy

Mapping is keyword-based over all report_0..report_17 columns in
machine_measurements.csv.  A subject receives a flag if ANY of their
ECG records contains a matching label.

Produces  <data_root>/mimic_superclass.csv
  columns: subject_id, NORM, MI, STTC, CD, HYP

Usage:
    python generate_mimic_superclass.py --data_root /path/to/mimic_ecg_project
"""
import argparse
import os
import re

import numpy as np
import pandas as pd


# ── Mapping rules ────────────────────────────────────────────────────────────
# Each entry is (superclass, match_fn) where match_fn(label: str) -> bool.
# Labels are already normalised (lowercase, stripped) before matching.

def _any(*patterns):
    """Return a function that is True if ANY pattern is found in the label."""
    compiled = [re.compile(p) for p in patterns]
    def _match(label):
        return any(c.search(label) for c in compiled)
    return _match


RULES = [
    # ── NORM ─────────────────────────────────────────────────────────────────
    ("NORM", _any(
        r"^normal ecg$",
        r"^normal ecg except for rate$",
        r"^normal ecg based on available leads$",
        r"^within normal limits$",
        r"^summary: normal ecg$",
        r"^no other finding$",
    )),

    # ── MI ───────────────────────────────────────────────────────────────────
    ("MI", _any(
        r"infarct",                                      # all infarct labels
        r"consider acute st elevation mi",               # *** consider acute st elevation mi ***
        r"st elevation, consider acute infarct",         # inferior/lateral/anteroseptal
        r"cannot rule out anteroseptal infarct",
        r"cannot rule out anterior infarct",
        r"cannot rule out septal infarct",
    )),

    # ── STTC ─────────────────────────────────────────────────────────────────
    ("STTC", _any(
        r"t wave change",
        r"st-t change",
        r"st change",                                    # lateral st changes are nonspecific
        r"st junctional",
        r"st elev",                                      # st elev, probable normal early repol
        r"prolonged qt",
        r"long qtc",
        r"tall t waves",
        r"repolarization change",
        r"borderline abnormal changes possibly due to myocardial ischemi",
        r"nonspecific t abnormali",
        r"borderline t abnormali",                       # borderline t abnormalities, diffuse leads
        r"abnormal t, consider ischemi",                 # abnormal t, consider ischemia, ...
        r"abnrm t, consider ischemi",                    # abbrev variant
        r"minimal st depression",
        r"st-t changes suggest myocardial injury",
        r"st-t changes may be due to myocardial ischemi",
        r"t wave changes may be due to myocardial ischemi",
        r"st-t changes are probably due to ventricular hypertrophy",
        r"t wave changes are probably due to ventricular hypertrophy",
        r"st-t changes may be due to hypertrophy",
        r"st-t changes are nonspecific",
        r"t changes are nonspecific",
        r"these minor changes are of equivocal significance",
    )),

    # ── CD ───────────────────────────────────────────────────────────────────
    ("CD", _any(
        r"bundle branch block",
        r"incomplete rbbb",
        r"incomplete lbbb",
        r"incomplete right bundle branch block",
        r"rbbb",                                         # rbbb and lafb, rbbb with lafb
        r"fascicular block",
        r"left anterior fascicular block",
        r"a-v block",
        r"a-v dissociation",
        r"pr interval",                                  # prolonged/borderline/short
        r"conduction defect",
        r"conduction delay",
        r"atrial fibrillation",
        r"atrial flutter",
        r"pacemaker",
        r"pacing",                                       # demand pacing, ventricular pacing, etc.
        r"pac\(s\)",
        r"pvc\(s\)",
        r"\bpacs\b",
        r"\bpvcs\b",
        r"premature ventricular complex",
        r"premature ventricular contraction",
        r"ventricular premature",
        r"atrial premature complex",
        r"supraventricular extrasystol",
        r"junctional rhythm",
        r"accelerated idioventricular",
        r"ectopic atrial",
        r"supraventricular rhythm",
        r"sinus or ectopic atrial rhythm",
        r"undetermined rhythm",
        r"ventricular-paced",                            # ventricular-paced rhythm/complexes
        r"atrial-sensed ventricular-paced",
        r"afib/flut",                                    # afib/flut and v-paced complexes
        r"ventricular bigeminy",
        r"ventricular trigeminy",
        r"supraventricular bigeminy",
        r"multiple premature complexes",
        r"axis deviation",
        r"leftward axis",
        r"rightward axis",
        r"indeterminate axis",
        r"marked left axis",
        r"severe right axis",
        r"left axis",
        r"right axis",
        r"wolff-parkinson-white",
        r"\bwpw\b",
        r"frequent pvcs",
        r"- first degree a-v block",
        r"- borderline first degree a-v block",
        r"- premature ventricular",
        r"- supraventricular",
    )),

    # ── HYP ──────────────────────────────────────────────────────────────────
    ("HYP", _any(
        r"hypertrophy",
        r"\blvh\b",
        r"atrial enlargement",
        r"atrial abnormality",
        r"low qrs voltage",
        r"low voltage",
        r"generalized low qrs",
        r"possible left atrial",
        r"probable left atrial",
        r"consider left atrial",
        r"left atrial",
        r"right atrial",
        r"st-t changes are probably due to ventricular hypertrophy",
        r"t wave changes are probably due to ventricular hypertrophy",
        r"st-t changes may be due to hypertrophy",
        r"qrs changes.*may be due to lvh",
    )),
]

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def classify_label(label: str) -> list[str]:
    """Return list of superclasses matched by a single normalised label string."""
    classes = []
    for sc, fn in RULES:
        if fn(label):
            classes.append(sc)
    return classes


def normalise(s: str) -> str:
    return s.strip().lower().rstrip(".").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--record_list_file", type=str, default="record_list.csv")
    parser.add_argument("--machine_measurements_file", type=str,
                        default="machine_measurements.csv")
    args = parser.parse_args()

    report_cols = [f"report_{i}" for i in range(18)]

    # ── Load ──────────────────────────────────────────────────────────────────
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
    print(f"Total ECG records: {len(df):,}  |  Subjects: {df['subject_id'].nunique():,}")

    # ── Normalise labels ──────────────────────────────────────────────────────
    for col in report_cols:
        df[col] = df[col].fillna("").astype(str).map(normalise)

    # ── Melt to long, classify each label ────────────────────────────────────
    label_long = (
        df.melt(id_vars="subject_id", value_vars=report_cols, value_name="label")
        .drop(columns="variable")
        .query("label != ''")
        .drop_duplicates(["subject_id", "label"])
    )

    # Expand to (subject_id, superclass) rows
    records_list = []
    for row in label_long.itertuples(index=False):
        for sc in classify_label(row.label):
            records_list.append((row.subject_id, sc))

    hits = pd.DataFrame(records_list, columns=["subject_id", "superclass"])
    hits = hits.drop_duplicates()

    # ── Pivot to wide binary flags ────────────────────────────────────────────
    subjects = sorted(df["subject_id"].dropna().unique())
    out = pd.DataFrame({"subject_id": subjects})
    for sc in SUPERCLASSES:
        flagged = set(hits.loc[hits["superclass"] == sc, "subject_id"])
        out[sc] = out["subject_id"].isin(flagged).astype(np.int8)

    # ── Stats ─────────────────────────────────────────────────────────────────
    n = len(out)
    print(f"\nSuperclass prevalence ({n:,} subjects):")
    for sc in SUPERCLASSES:
        k = out[sc].sum()
        print(f"  {sc:4s}  {k:7,}  ({100*k/n:.1f}%)")

    unclassified = (out[SUPERCLASSES].sum(axis=1) == 0).sum()
    print(f"\n  Unclassified (no superclass): {unclassified:,}  ({100*unclassified/n:.1f}%)")

    overlap = out[SUPERCLASSES].sum(axis=1)
    print(f"\n  Labels per subject: mean={overlap.mean():.2f}  "
          f"median={overlap.median():.0f}  max={overlap.max():.0f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.data_root, "mimic_superclass.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
