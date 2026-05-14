"""Build ECG-level diagnostic labels from MIMIC machine reports.

Parses the 18 report fields in machine_measurements.csv and maps each
statement to PTB-XL-compatible SCP codes using regex rules. This gives
ECG-level ground truth (what THIS ECG shows) rather than patient-level
ICD discharge codes.

Labels are aligned with PTB-XL's scp_statements.csv so that:
  - The same label names are used (AFIB, IMI, LVH, CLBBB, etc.)
  - Each label maps to a diagnostic_class (superclass): NORM, MI, STTC, CD, HYP
  - Models trained on MIMIC can be evaluated on PTB-XL and vice versa

Usage
-----
    python prepare_ecg_labels.py \
        --machine_csv /path/to/machine_measurements.csv \
        --record_list /path/to/record_list.csv \
        --out /path/to/mimic_ecg_labels.csv \
        --level subclass        # subclass (fine-grained) or superclass (5 classes)
"""

import argparse
import re

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Mapping: MIMIC report text → PTB-XL SCP code → superclass
#
# Source: PTB-XL scp_statements.csv
# Each entry: (scp_code, superclass, [regex patterns matching MIMIC text])
# ═══════════════════════════════════════════════════════════════════════

LABEL_RULES = [
    # ── NORM ────────────────────────────────────────────────────────
    # PTB-XL: NORM = normal ECG (diagnostic_class=NORM)
    ("NORM", "NORM", [
        r"^normal ecg$",
        r"^normal ecg except for rate$",
        r"^normal ecg based on available leads$",
        r"^sinus rhythm$",
        r"^sinus rhythm\.$",
    ]),

    # ── MI (Myocardial Infarction) ──────────────────────────────────
    # PTB-XL: IMI = inferior MI
    ("IMI", "MI", [
        r"inferior infarct",
    ]),
    # PTB-XL: ASMI = anteroseptal MI
    ("ASMI", "MI", [
        r"anteroseptal infarct",
    ]),
    # PTB-XL: AMI = anterior MI
    ("AMI", "MI", [
        r"anterior infarct(?!.*septal)",
    ]),
    # PTB-XL: ALMI = anterolateral MI
    ("ALMI", "MI", [
        r"anterolateral infarct",
    ]),
    # PTB-XL: LMI = lateral MI
    ("LMI", "MI", [
        r"lateral infarct(?!.*(?:anterior|inferior|postero))",
    ]),
    # PTB-XL: ILMI = inferolateral MI
    ("ILMI", "MI", [
        r"inferolateral infarct",
    ]),
    # PTB-XL: PMI = posterior MI
    ("PMI", "MI", [
        r"posterior infarct",
    ]),
    # PTB-XL: IPMI = inferoposterior MI
    ("IPMI", "MI", [
        r"inferoposterior infarct",
    ]),
    # No direct PTB-XL code, but MI-class:
    ("EXTENSIVE_MI", "MI", [
        r"extensive infarct",
    ]),
    ("SEPTAL_MI", "MI", [
        r"septal infarct(?!.*(?:anterior|antero))",
    ]),
    # PTB-XL: INJAS/INJAL/INJIN/INJLA/INJIL = subendocardial injury
    ("ACUTE_MI", "MI", [
        r"consider acute (?:st elevation )?(?:mi|infarct)",
        r"\*\*\* consider acute st elevation mi \*\*\*",
        r"acute (?:lateral |inferior |anteroseptal )?infarct",
        r"st elevation.*consider acute infarct",
    ]),

    # ── STTC (ST/T Changes) ────────────────────────────────────────
    # PTB-XL: ISC_ = non-specific ischemic ST-T changes
    ("ISC_", "STTC", [
        r"myocardial ischemia",
        r"ischemi[ac]",
        r"st-t changes (?:may be|are probably) due to (?:myocardial )?ischemia",
    ]),
    # PTB-XL: NST_ = non-specific ST changes
    ("NST_", "STTC", [
        r"st-t changes (?:are |is )?nonspecific",
        r"st changes (?:are |is )?nonspecific",
        r"st-t changes$",
    ]),
    # PTB-XL: NDT = non-diagnostic T abnormalities
    ("NDT", "STTC", [
        r"t wave changes (?:are |is )?nonspecific",
        r"t (?:wave )?changes (?:may be )?normal for age",
        r"nonspecific t (?:wave )?abnormalit",
    ]),
    # PTB-XL: STD_ = non-specific ST depression
    ("STD_", "STTC", [
        r"st (?:junctional )?depression",
    ]),
    # PTB-XL: STE_ = non-specific ST elevation
    ("STE_", "STTC", [
        r"st elev(?:ation)?(?!.*(?:normal early repol|probable normal|consider acute|mi))",
    ]),
    # PTB-XL: LNGQT = long QT interval
    ("LNGQT", "STTC", [
        r"prolonged qt(?:c)? interval",
        r"long qtc",
    ]),
    # PTB-XL: INVT = inverted T-waves
    ("INVT", "STTC", [
        r"inverted t.wave",
    ]),
    # PTB-XL: ANEUR = ST-T changes compatible with ventricular aneurysm
    ("ANEUR", "STTC", [
        r"ventricular aneurysm",
    ]),
    # PTB-XL: TAB_ = T-wave abnormality (catch-all for remaining T changes)
    ("TAB_", "STTC", [
        r"t wave changes (?:may be due to|are probably due to) (?:ventricular )?hypertrophy",
    ]),
    # Early repolarization (not in PTB-XL SCP but STTC-class)
    ("EARLY_REPOL", "STTC", [
        r"early repol(?:arization)?",
        r"probable normal early repol",
    ]),

    # ── CD (Conduction Disturbance) ─────────────────────────────────
    # PTB-XL: CLBBB = complete LBBB
    ("CLBBB", "CD", [
        r"(?:complete )?left bundle branch block",
        r"\blbbb\b",
    ]),
    # PTB-XL: ILBBB = incomplete LBBB
    ("ILBBB", "CD", [
        r"incomplete l(?:eft )?b(?:undle )?b(?:ranch )?b(?:lock)?",
    ]),
    # PTB-XL: CRBBB = complete RBBB
    ("CRBBB", "CD", [
        r"(?:complete )?right bundle branch block",
        r"\brbbb\b(?! with| and)",
    ]),
    # PTB-XL: IRBBB = incomplete RBBB
    ("IRBBB", "CD", [
        r"incomplete r(?:ight )?b(?:undle )?b(?:ranch )?b(?:lock)?",
    ]),
    # PTB-XL: LAFB
    ("LAFB", "CD", [
        r"left anterior fascicular block",
        r"rbbb (?:with|and) (?:left anterior fascicular block|lafb)",
        r"rbbb and lafb",
    ]),
    # PTB-XL: LPFB
    ("LPFB", "CD", [
        r"left posterior fascicular block",
    ]),
    # PTB-XL: IVCD
    ("IVCD", "CD", [
        r"(?:nonspecific )?intraventricular conduction (?:delay|defect)",
        r"iv conduction defect",
    ]),
    # PTB-XL: 1AVB
    ("1AVB", "CD", [
        r"1st degree a-v block",
        r"prolonged pr interval",
        r"borderline (?:prolonged pr|1st degree)",
    ]),
    # PTB-XL: 2AVB
    ("2AVB", "CD", [
        r"2nd degree (?:a-v|heart) block",
    ]),
    # PTB-XL: 3AVB
    ("3AVB", "CD", [
        r"(?:3rd degree|complete) (?:a-v|heart) block",
    ]),
    # PTB-XL: WPW
    ("WPW", "CD", [
        r"wolff.parkinson.white",
        r"\bwpw\b",
        r"pre.excitation",
    ]),
    # Short PR (not in PTB-XL diagnostic codes, but CD-class)
    ("SHORT_PR", "CD", [
        r"short pr interval",
    ]),

    # ── HYP (Hypertrophy) ──────────────────────────────────────────
    # PTB-XL: LVH
    ("LVH", "HYP", [
        r"left ventricular hypertrophy",
        r"\blvh\b",
        r"voltage criteria.*left ventricular hypertrophy",
    ]),
    # PTB-XL: RVH
    ("RVH", "HYP", [
        r"right ventricular hypertrophy",
    ]),
    # PTB-XL: LAO/LAE
    ("LAO/LAE", "HYP", [
        r"left atrial (?:enlargement|abnormality|overload)",
        r"probable left atrial enlargement",
        r"possible left atrial (?:abnormality|enlargement)",
        r"consider left atrial abnormality",
    ]),
    # PTB-XL: RAO/RAE
    ("RAO/RAE", "HYP", [
        r"right atrial (?:enlargement|abnormality|overload)",
        r"possible right atrial (?:abnormality|enlargement)",
    ]),

    # ── Rhythm (PTB-XL rhythm codes — not diagnostic_class but useful) ──
    # PTB-XL: AFIB
    ("AFIB", None, [
        r"atrial fibrillation",
    ]),
    # PTB-XL: AFLT
    ("AFLT", None, [
        r"atrial flutter",
    ]),
    # PTB-XL: SBRAD
    ("SBRAD", None, [
        r"sinus bradycardia",
    ]),
    # PTB-XL: STACH
    ("STACH", None, [
        r"sinus tachycardia",
    ]),
    # PTB-XL: SARRH
    ("SARRH", None, [
        r"sinus arrhythmia",
    ]),
    # PTB-XL: SVTAC / SVARR
    ("SVTAC", None, [
        r"supraventricular (?:rhythm|tachycardia)",
    ]),
    # PTB-XL: PACE
    ("PACE", None, [
        r"pacemaker rhythm",
        r"demand (?:atrial )?pacing",
        r"ventricular pacing",
        r"a-v sequential pacemaker",
        r"atrial pacing",
    ]),
    # PTB-XL: PAC
    ("PAC", None, [
        r"atrial premature complex",
        r"\bpac\b",
        r"premature atrial",
    ]),
    # PTB-XL: PVC
    ("PVC", None, [
        r"ventricular premature complex",
        r"\bpvc\b",
        r"premature ventricular contraction",
    ]),

    # ── Form codes (PTB-XL form codes — not diagnostic_class) ──────
    # PTB-XL: LVOLT
    ("LVOLT", None, [
        r"low (?:qrs )?voltage",
        r"generalized low",
    ]),
    # PTB-XL: QWAVE
    ("QWAVE", None, [
        r"pathological q.wave",
        r"abnormal q",
        r"q waves present",
    ]),
    # PTB-XL: ABQRS
    ("ABQRS", None, [
        r"abnormal qrs",
        r"poor r.wave progression",
    ]),
    # Axis deviations (not in PTB-XL SCP but useful)
    ("LEFT_AXIS", None, [
        r"left axis deviation",
        r"leftward axis",
        r"marked left axis",
    ]),
    ("RIGHT_AXIS", None, [
        r"right axis deviation",
        r"rightward axis",
        r"severe right axis",
    ]),
]


# Build: scp_code → superclass mapping
SCP_TO_SUPERCLASS = {}
for scp_code, superclass, _ in LABEL_RULES:
    if superclass is not None:
        SCP_TO_SUPERCLASS[scp_code] = superclass

# Compile regexes: list of (scp_code, compiled_pattern)
COMPILED_RULES = []
for scp_code, _, patterns in LABEL_RULES:
    for pattern in patterns:
        COMPILED_RULES.append((scp_code, re.compile(pattern, re.IGNORECASE)))


# Meta-statements to skip
SKIP_PATTERNS = re.compile(
    r"abnormal ecg|borderline ecg|summary:|"
    r"warning.*data quality|report made without knowing|"
    r"lead.*unsuitable|age not entered|"
    r"suspect arm lead reversal|"
    r"lateral leads are also involved|"
    r"changes may (?:also )?(?:partly )?be due to rhythm|"
    r"repolarization changes may be partly due",
    re.IGNORECASE,
)


def extract_labels(report_fields):
    """Extract PTB-XL-aligned SCP codes from MIMIC report fields.

    Returns a set of SCP code names found in this ECG's report.
    """
    labels = set()
    for field in report_fields:
        field = field.strip().rstrip(".")
        if not field:
            continue
        if SKIP_PATTERNS.search(field):
            if not re.match(r"^normal ecg", field, re.IGNORECASE):
                continue
        for scp_code, pattern in COMPILED_RULES:
            if pattern.search(field):
                labels.add(scp_code)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Build ECG-level labels from machine reports")
    parser.add_argument("--machine_csv", type=str, required=True,
                        help="Path to machine_measurements.csv")
    parser.add_argument("--record_list", type=str, default=None,
                        help="Path to record_list.csv (to filter to available waveforms)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum occurrences for a label to be included (default: 0 = keep all)")
    parser.add_argument("--level", type=str, default="subclass",
                        choices=["subclass", "superclass"],
                        help="Label granularity: 'subclass' = individual SCP codes (~30-40), "
                             "'superclass' = 5 PTB-XL classes (NORM/MI/STTC/CD/HYP)")
    args = parser.parse_args()

    # ── Load reports ──────────────────────────────────────────────
    print(f"Reading machine reports from: {args.machine_csv}")
    report_cols = [f"report_{i}" for i in range(18)]
    df = pd.read_csv(
        args.machine_csv,
        usecols=["study_id"] + report_cols,
        low_memory=False,
    )
    print(f"  Total rows: {len(df)}")

    for c in report_cols:
        df[c] = df[c].fillna("").astype(str).str.strip()

    # ── Extract labels ────────────────────────────────────────────
    print("  Extracting labels from reports...")
    all_report_fields = df[report_cols].values.tolist()
    extracted = [extract_labels(fields) for fields in all_report_fields]

    # ── Map to superclass if requested ────────────────────────────
    if args.level == "superclass":
        print("  Mapping to 5 PTB-XL superclasses (NORM, MI, STTC, CD, HYP)")
        mapped = []
        for labels in extracted:
            superclasses = set()
            for l in labels:
                sc = SCP_TO_SUPERCLASS.get(l)
                if sc is not None:
                    superclasses.add(sc)
                # If NORM is present with other findings, NORM is ambiguous
                # Keep it — the model should learn that NORM + something = not truly normal
            mapped.append(superclasses)
        extracted = mapped

    # ── Count label frequencies ───────────────────────────────────
    from collections import Counter
    label_counts = Counter()
    for labels in extracted:
        for l in labels:
            label_counts[l] += 1

    print(f"\n  All labels found ({len(label_counts)}):")
    for label, count in label_counts.most_common():
        sc = SCP_TO_SUPERCLASS.get(label, "-")
        marker = "*" if count >= args.min_count else " "
        print(f"    {marker} {label:15s} (superclass={sc:5s})  {count:>8} ({100*count/len(df):.1f}%)")

    # Filter to frequent labels
    kept_labels = sorted(l for l, n in label_counts.items() if n >= args.min_count)
    kept_set = set(kept_labels)
    print(f"\n  Labels with >= {args.min_count} occurrences: {len(kept_labels)}")

    # ── Filter to available waveforms ─────────────────────────────
    if args.record_list:
        print(f"  Filtering to available waveforms in: {args.record_list}")
        records = pd.read_csv(args.record_list, usecols=["study_id"])
        available = set(records["study_id"].unique())
        mask = df["study_id"].isin(available)
        df = df[mask].copy()
        extracted = [extracted[i] for i in range(len(mask)) if mask.values[i]]
        print(f"  Matched to waveforms: {len(df)}")

    # ── Filter extracted labels to kept set ───────────────────────
    extracted = [labels & kept_set for labels in extracted]

    has_labels = [len(labels) > 0 for labels in extracted]
    n_with = sum(has_labels)
    n_without = len(df) - n_with
    print(f"  Rows with >= 1 label: {n_with} (no labels: {n_without})")

    # ── Build output matrix ───────────────────────────────────────
    label_map = {label: i for i, label in enumerate(kept_labels)}
    n_rows = len(df)
    n_labels = len(kept_labels)

    print(f"\n  Building label matrix ({n_rows} x {n_labels}) ...")
    study_ids = df["study_id"].values.astype(int)
    label_matrix = np.zeros((n_rows, n_labels), dtype=np.int8)

    for i, labels in enumerate(extracted):
        for l in labels:
            label_matrix[i, label_map[l]] = 1

    n_with = (label_matrix.sum(axis=1) > 0).sum()
    print(f"  Rows with >= 1 label: {n_with} / {n_rows} "
          f"({n_rows - n_with} all-zero — kept for consistent sample set)")
    print(f"  Writing {n_rows} rows to: {args.out}")

    header = ["study_id"] + kept_labels
    with open(args.out, "w") as f:
        f.write(",".join(header) + "\n")
        chunk_size = 50000
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            lines = []
            for i in range(start, end):
                row_str = f"{study_ids[i]}," + ",".join(label_matrix[i].astype(str))
                lines.append(row_str)
            f.write("\n".join(lines) + "\n")
            print(f"    wrote rows {start}-{end}")

    print(f"\n  Done: {n_rows} ECGs, {n_labels} labels")
    print(f"  Labels: {kept_labels}")

    # ── Save label map with superclass info ───────────────────────
    map_path = args.out.replace(".csv", "_label_map.csv")
    map_df = pd.DataFrame([{
        "index": i,
        "scp_code": label,
        "superclass": SCP_TO_SUPERCLASS.get(label, ""),
        "count": label_counts[label],
    } for label, i in label_map.items()])
    map_df.to_csv(map_path, index=False)
    print(f"  Label map saved to: {map_path}")


if __name__ == "__main__":
    main()
