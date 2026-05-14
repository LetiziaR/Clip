"""
Generate presentation-quality figures for CoCa ECG-to-Text project.

Outputs (saved to figures/ directory):
  1. fig_ablation_table.pdf     — Ablation study table
  2. fig_retrieval.pdf          — R@1 vs random baseline
  3. fig_generation.pdf         — Generation examples + quantitative scores
  4. fig_training_curves.pdf    — Convergence proof (loss + retrieval)
"""

import os
import json
import glob
import textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# ── Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

CHECKPOINT_DIR = "checkpoints"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# ── Experiment groups for ablation ──
# Only include the main sweep experiments (not legacy coca_* / ts2vec_*)
MAIN_EXPERIMENTS = {
    "bart", "biobart", "clinical_t5", "gpt2", "biogpt",
    "bart_pseudo", "patchtst_bart",
    "bart_lr1e4", "bart_lr1e5",
    "bart_temp01", "bart_temp005",
    "bart_with_cls", "bart_bs64", "bart_proj256",
}

# For training curves, show only a readable subset
CURVE_EXPERIMENTS = [
    "bart", "biobart", "biogpt", "clinical_t5", "gpt2",
    "bart_proj256", "patchtst_bart",
]

CURVE_COLORS = {
    "bart": "#2196F3",
    "biobart": "#4CAF50",
    "biogpt": "#FF9800",
    "clinical_t5": "#9C27B0",
    "gpt2": "#F44336",
    "bart_proj256": "#00BCD4",
    "patchtst_bart": "#795548",
}

# ── Load Data ──
def load_runs():
    runs = {}
    for run_dir in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*"))):
        name = os.path.basename(run_dir)
        if name not in MAIN_EXPERIMENTS:
            continue
        config_path = os.path.join(run_dir, "config.json")
        summary_path = os.path.join(run_dir, "summary.json")
        metrics_path = os.path.join(run_dir, "metrics.csv")
        gen_metrics_path = os.path.join(run_dir, "eval_generation", "generation_metrics.json")

        if not os.path.isfile(metrics_path):
            continue

        config = json.load(open(config_path)) if os.path.isfile(config_path) else {}
        summary = json.load(open(summary_path)) if os.path.isfile(summary_path) else {}
        metrics = pd.read_csv(metrics_path)
        gen_metrics = json.load(open(gen_metrics_path)) if os.path.isfile(gen_metrics_path) else {}

        runs[name] = {
            "config": config,
            "summary": summary,
            "metrics": metrics,
            "gen_metrics": gen_metrics,
        }
    return runs


def best_row(metrics_df):
    """Return the row with lowest val_loss."""
    idx = metrics_df["val_loss"].idxmin()
    return metrics_df.iloc[idx]


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: ABLATION TABLE
# ═══════════════════════════════════════════════════════════════════════
def fig_ablation_table(runs):
    """Two-panel ablation table: decoder comparison + hyperparameter/design ablations."""

    # ── Decoder comparison (vs bart baseline) ──
    decoder_names = ["bart", "biobart", "clinical_t5", "gpt2", "biogpt"]
    decoder_labels = ["BART (baseline)", "BioBart", "Clinical-T5", "GPT-2", "BioGPT"]

    # ── Ablation rows (each varies one thing from bart baseline) ──
    ablation_specs = [
        # (exp_name, display_label, what_changed)
        ("bart",          "BART (baseline)", "—"),
        ("bart_proj256",  "Proj. dim 256",   "projection_dim: 128→256"),
        ("bart_temp01",   "Temp. 0.10",      "temperature: 0.07→0.10"),
        ("bart_temp005",  "Temp. 0.05",      "temperature: 0.07→0.05"),
        ("bart_lr1e4",    "LR 1e-4",         "learning_rate: 5e-5→1e-4"),
        ("bart_lr1e5",    "LR 1e-5",         "learning_rate: 5e-5→1e-5"),
        ("bart_with_cls", "+ Cls. head",      "classification_head: on"),
        ("bart_bs64",     "Batch size 64",    "batch_size: 32→64"),
        ("bart_pseudo",   "Pseudo-reports",   "text_source: pseudo"),
        ("patchtst_bart", "PatchTST encoder", "ts_encoder: PatchTST"),
    ]

    # Helper to extract metrics
    def row_data(name):
        if name not in runs:
            return None
        br = best_row(runs[name]["metrics"])
        gm = runs[name].get("gen_metrics", {})
        return {
            "val_loss": br["val_loss"],
            "caption": br["val_caption"],
            "contrastive": br["val_contrastive"],
            "R@1": br.get("val_ecg2text_R@1", 0) * 100,
            "R@5": br.get("val_ecg2text_R@5", 0) * 100,
            "ROUGE-L": gm.get("rougeL_f1", 0) * 100,
            "BLEU-1": gm.get("bleu1", 0) * 100,
            "ClinF1": gm.get("clinical_concept_f1", 0) * 100,
        }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5),
                                     gridspec_kw={"height_ratios": [1, 1.6]})

    # ── Panel A: Decoder comparison ──
    cols_a = ["Decoder", "Val Loss ↓", "Caption ↓", "Contr. ↓",
              "R@1 (%) ↑", "R@5 (%) ↑", "ROUGE-L ↑", "BLEU-1 ↑", "Clin-F1 ↑"]
    cell_data_a = []
    for name, label in zip(decoder_names, decoder_labels):
        d = row_data(name)
        if d is None:
            continue
        cell_data_a.append([
            label,
            f"{d['val_loss']:.3f}",
            f"{d['caption']:.3f}",
            f"{d['contrastive']:.3f}",
            f"{d['R@1']:.2f}",
            f"{d['R@5']:.2f}",
            f"{d['ROUGE-L']:.1f}",
            f"{d['BLEU-1']:.1f}",
            f"{d['ClinF1']:.1f}",
        ])

    ax1.axis("off")
    ax1.set_title("(a) Decoder Architecture Comparison", fontweight="bold", loc="left", pad=10)
    table_a = ax1.table(
        cellText=cell_data_a, colLabels=cols_a,
        loc="center", cellLoc="center",
    )
    table_a.auto_set_font_size(False)
    table_a.set_fontsize(9.5)
    table_a.scale(1, 1.45)

    # Style header
    for j in range(len(cols_a)):
        table_a[0, j].set_facecolor("#37474F")
        table_a[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight baseline row
    for j in range(len(cols_a)):
        table_a[1, j].set_facecolor("#E3F2FD")

    # Highlight best values (skip first column)
    metrics_cols = list(range(1, len(cols_a)))
    # For losses (cols 1-3): lower is better → highlight min
    # For scores (cols 4-8): higher is better → highlight max
    for col_idx in metrics_cols:
        vals = []
        for row_idx in range(len(cell_data_a)):
            try:
                vals.append(float(cell_data_a[row_idx][col_idx]))
            except ValueError:
                vals.append(None)
        valid = [v for v in vals if v is not None]
        if not valid:
            continue
        if col_idx <= 3:  # loss columns
            best_val = min(valid)
        else:  # score columns
            best_val = max(valid)
        for row_idx, v in enumerate(vals):
            if v == best_val:
                table_a[row_idx + 1, col_idx].set_text_props(fontweight="bold", color="#1B5E20")

    # ── Panel B: Hyperparameter & design ablations ──
    cols_b = ["Variant", "Changed", "Val Loss ↓", "Caption ↓", "Contr. ↓",
              "R@1 (%) ↑", "R@5 (%) ↑", "ROUGE-L ↑", "BLEU-1 ↑"]
    cell_data_b = []
    for name, label, change in ablation_specs:
        d = row_data(name)
        if d is None:
            continue
        cell_data_b.append([
            label, change,
            f"{d['val_loss']:.3f}",
            f"{d['caption']:.3f}",
            f"{d['contrastive']:.3f}",
            f"{d['R@1']:.2f}",
            f"{d['R@5']:.2f}",
            f"{d['ROUGE-L']:.1f}",
            f"{d['BLEU-1']:.1f}",
        ])

    ax2.axis("off")
    ax2.set_title("(b) Hyperparameter & Design Ablations (vs BART baseline)",
                   fontweight="bold", loc="left", pad=10)
    table_b = ax2.table(
        cellText=cell_data_b, colLabels=cols_b,
        loc="center", cellLoc="center",
    )
    table_b.auto_set_font_size(False)
    table_b.set_fontsize(9.5)
    table_b.scale(1, 1.35)

    for j in range(len(cols_b)):
        table_b[0, j].set_facecolor("#37474F")
        table_b[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight baseline row
    for j in range(len(cols_b)):
        table_b[1, j].set_facecolor("#E3F2FD")

    # Color-code improvements vs baseline
    if cell_data_b:
        baseline = cell_data_b[0]  # BART baseline
        for row_idx in range(1, len(cell_data_b)):
            for col_idx in range(2, len(cols_b)):
                try:
                    val = float(cell_data_b[row_idx][col_idx])
                    base = float(baseline[col_idx])
                except (ValueError, IndexError):
                    continue
                if col_idx <= 4:  # loss: lower is better
                    if val < base - 0.005:
                        table_b[row_idx + 1, col_idx].set_text_props(
                            fontweight="bold", color="#1B5E20")
                    elif val > base + 0.02:
                        table_b[row_idx + 1, col_idx].set_text_props(color="#B71C1C")
                else:  # scores: higher is better
                    if val > base + 0.1:
                        table_b[row_idx + 1, col_idx].set_text_props(
                            fontweight="bold", color="#1B5E20")
                    elif val < base - 0.5:
                        table_b[row_idx + 1, col_idx].set_text_props(color="#B71C1C")

    plt.tight_layout(h_pad=2.0)
    path = os.path.join(OUTDIR, "fig_ablation_table.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: RETRIEVAL R@1 vs RANDOM BASELINE
# ═══════════════════════════════════════════════════════════════════════
def fig_retrieval(runs):
    """Bar chart of best R@1 per experiment, with random baseline annotation."""

    VAL_SIZE = 2184  # PTB-XL fold 9
    random_r1 = 100.0 / VAL_SIZE  # ~0.046%

    # Collect best R@1 per experiment
    results = []
    for name in sorted(MAIN_EXPERIMENTS):
        if name not in runs:
            continue
        br = best_row(runs[name]["metrics"])
        r1 = br.get("val_ecg2text_R@1", 0) * 100
        r5 = br.get("val_ecg2text_R@5", 0) * 100
        results.append({"name": name, "R@1": r1, "R@5": r5})

    df = pd.DataFrame(results).sort_values("R@1", ascending=True)

    # Pretty labels
    label_map = {
        "bart": "BART", "biobart": "BioBart", "biogpt": "BioGPT",
        "clinical_t5": "Clinical-T5", "gpt2": "GPT-2",
        "bart_proj256": "BART (proj=256)", "bart_temp01": "BART (τ=0.10)",
        "bart_temp005": "BART (τ=0.05)", "bart_lr1e4": "BART (lr=1e-4)",
        "bart_lr1e5": "BART (lr=1e-5)", "bart_with_cls": "BART (+cls)",
        "bart_bs64": "BART (bs=64)", "bart_pseudo": "BART (pseudo)",
        "patchtst_bart": "PatchTST+BART",
    }
    df["label"] = df["name"].map(label_map)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color bars by performance tier
    colors = []
    for _, row in df.iterrows():
        if row["R@1"] >= 3.5:
            colors.append("#2E7D32")   # best tier
        elif row["R@1"] >= 2.5:
            colors.append("#1976D2")   # good
        elif row["R@1"] >= 1.5:
            colors.append("#F57C00")   # moderate
        else:
            colors.append("#C62828")   # poor

    bars = ax.barh(df["label"], df["R@1"], color=colors, edgecolor="white", height=0.65)

    # Random baseline
    ax.axvline(x=random_r1, color="#D32F2F", linestyle="--", linewidth=1.5, zorder=5)
    ax.text(random_r1 + 0.08, len(df) - 0.5,
            f"Random chance\n({random_r1:.3f}%)",
            color="#D32F2F", fontsize=9, fontweight="bold", va="top")

    # Annotate bars with value and multiplier over random
    for bar, (_, row) in zip(bars, df.iterrows()):
        mult = row["R@1"] / random_r1
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                f'{row["R@1"]:.2f}%  ({mult:.0f}× random)',
                va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("ECG→Text Recall@1 (%)", fontweight="bold")
    ax.set_title("Retrieval Performance: ECG→Text R@1", fontweight="bold", fontsize=14)
    ax.set_xlim(0, df["R@1"].max() * 1.45)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for color tiers
    legend_elements = [
        mpatches.Patch(facecolor="#2E7D32", label="R@1 ≥ 3.5%"),
        mpatches.Patch(facecolor="#1976D2", label="R@1 ≥ 2.5%"),
        mpatches.Patch(facecolor="#F57C00", label="R@1 ≥ 1.5%"),
        mpatches.Patch(facecolor="#C62828", label="R@1 < 1.5%"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    # Add subtitle
    fig.text(0.5, -0.02,
             f"Val set: {VAL_SIZE} samples  |  Random baseline: 1/{VAL_SIZE} = {random_r1:.3f}%  |  "
             f"Best model achieves {df['R@1'].max():.2f}% ({df['R@1'].max()/random_r1:.0f}× random)",
             ha="center", fontsize=10, style="italic", color="#555")

    path = os.path.join(OUTDIR, "fig_retrieval.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: GENERATION EXAMPLES (qualitative + quantitative)
# ═══════════════════════════════════════════════════════════════════════
def fig_generation(runs):
    """Combined figure: quantitative scores table + qualitative examples."""

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.8], hspace=0.35)

    # ── Panel A: Quantitative generation metrics ──
    ax_quant = fig.add_subplot(gs[0])
    ax_quant.axis("off")
    ax_quant.set_title("(a) Text Generation Quality (test set, n=2,198)",
                       fontweight="bold", loc="left", pad=10)

    # Order experiments by ROUGE-L
    gen_exps = []
    for name in ["bart", "biobart", "bart_proj256", "bart_with_cls", "bart_lr1e4",
                 "bart_bs64", "bart_temp01", "bart_temp005", "bart_lr1e5",
                 "bart_pseudo", "patchtst_bart", "gpt2", "biogpt"]:
        if name in runs and runs[name].get("gen_metrics"):
            gen_exps.append(name)

    label_map = {
        "bart": "BART", "biobart": "BioBart", "biogpt": "BioGPT",
        "clinical_t5": "Clin-T5", "gpt2": "GPT-2",
        "bart_proj256": "BART (proj=256)", "bart_temp01": "BART (τ=0.10)",
        "bart_temp005": "BART (τ=0.05)", "bart_lr1e4": "BART (lr=1e-4)",
        "bart_lr1e5": "BART (lr=1e-5)", "bart_with_cls": "BART (+cls)",
        "bart_bs64": "BART (bs=64)", "bart_pseudo": "BART (pseudo)",
        "patchtst_bart": "PatchTST+BART",
    }

    cols_q = ["Model", "ROUGE-L", "BLEU-1", "BLEU-2", "METEOR",
              "BERTScore", "Clin-Prec", "Clin-Rec", "Clin-F1"]
    cell_q = []
    for name in gen_exps:
        gm = runs[name]["gen_metrics"]
        cell_q.append([
            label_map.get(name, name),
            f"{gm['rougeL_f1']*100:.1f}",
            f"{gm['bleu1']*100:.1f}",
            f"{gm['bleu2']*100:.1f}",
            f"{gm['meteor']*100:.1f}",
            f"{gm['bertscore_f1']*100:.1f}",
            f"{gm['clinical_concept_precision']*100:.1f}",
            f"{gm['clinical_concept_recall']*100:.1f}",
            f"{gm['clinical_concept_f1']*100:.1f}",
        ])

    # Sort by ROUGE-L descending
    cell_q.sort(key=lambda r: -float(r[1]))

    table_q = ax_quant.table(
        cellText=cell_q, colLabels=cols_q,
        loc="center", cellLoc="center",
    )
    table_q.auto_set_font_size(False)
    table_q.set_fontsize(8.5)
    table_q.scale(1, 1.3)

    for j in range(len(cols_q)):
        table_q[0, j].set_facecolor("#37474F")
        table_q[0, j].set_text_props(color="white", fontweight="bold", fontsize=8.5)

    # Bold the best value in each score column
    for col_idx in range(1, len(cols_q)):
        vals = [float(row[col_idx]) for row in cell_q]
        # BERTScore can be negative for bad models, still highlight max
        best_val = max(vals)
        for row_idx, v in enumerate(vals):
            if v == best_val:
                table_q[row_idx + 1, col_idx].set_text_props(
                    fontweight="bold", color="#1B5E20")

    # ── Panel B: Qualitative examples ──
    ax_qual = fig.add_subplot(gs[1])
    ax_qual.axis("off")
    ax_qual.set_title("(b) Generation Examples (BART model)",
                       fontweight="bold", loc="left", pad=10)

    # Curated examples showing different qualities
    examples = [
        # Perfect match
        {"idx": 2, "verdict": "Exact match",
         "pred": "sinusrhythmus linkstyp sonst normales ekg",
         "ref":  "sinusrhythmus linkstyp sonst normales ekg"},
        # Near-match: correct rhythm, captures abnormality
        {"idx": 263, "verdict": "Near match",
         "pred": "premature atrial contraction(s). sinus rhythm. voltages are high\n"
                 "in limb leads suggesting lvh. st segments are depressed in i, avl, v4,5,6.\n"
                 "t waves are low or flat. this may be due to lv strain or ischaemia",
         "ref":  "premature atrial contraction(s). sinus rhythm. left axis deviation.\n"
                 "left ventricular hypertrophy. st segments are depressed in i, avl, v4,5,6.\n"
                 "t waves are flat in i and inverted in avl, v4,5,6. this may be due to\n"
                 "lv strain or ischaemia"},
        # Partially correct: gets sinus rhythm right but misses pathology
        {"idx": 292, "verdict": "Partial",
         "pred": "sinus rhythm. no definite pathology",
         "ref":  "sinus rhythm. left anterior fascicular block.\n"
                 "otherwise no definite pathology"},
        # Interesting: correct rhythm, hallucinated details
        {"idx": 81, "verdict": "Correct core,\nwrong rhythm",
         "pred": "sinus rhythm. left axis deviation. left bundle branch block,\n"
                 "this is most commonly due to ischaemic heart disease",
         "ref":  "atrial fibrillation. left axis deviation. left bundle branch block,\n"
                 "this is most commonly due to ischaemic heart disease"},
        # Failure: misses complex pathology
        {"idx": 19, "verdict": "Miss",
         "pred": "sinusrhythmus lagetyp normal unvollständiger rechtsschenkelblock\n"
                 "sonst normales ekg",
         "ref":  "ventrikuläre extrasystole(n) interponierte ventrikuläre\n"
                 "extrasystole(n) sinus arrhythmie unspezifisches abnormales t"},
    ]

    cols_e = ["#", "Quality", "Generated Report (Prediction)", "Ground Truth (Reference)"]
    cell_e = []
    for ex in examples:
        cell_e.append([
            str(ex["idx"]),
            ex["verdict"],
            ex["pred"],
            ex["ref"],
        ])

    table_e = ax_qual.table(
        cellText=cell_e, colLabels=cols_e,
        loc="center", cellLoc="left",
        colWidths=[0.04, 0.09, 0.435, 0.435],
    )
    table_e.auto_set_font_size(False)
    table_e.set_fontsize(8)
    table_e.scale(1, 2.2)

    for j in range(len(cols_e)):
        table_e[0, j].set_facecolor("#37474F")
        table_e[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)

    # Color quality column
    quality_colors = {
        "Exact match": "#C8E6C9",
        "Near match": "#DCEDC8",
        "Partial": "#FFF9C4",
        "Correct core,\nwrong rhythm": "#FFE0B2",
        "Miss": "#FFCDD2",
    }
    for row_idx, ex in enumerate(examples):
        color = quality_colors.get(ex["verdict"], "white")
        table_e[row_idx + 1, 1].set_facecolor(color)
        table_e[row_idx + 1, 1].set_text_props(fontweight="bold", fontsize=8)

    path = os.path.join(OUTDIR, "fig_generation.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: TRAINING CURVES (convergence proof)
# ═══════════════════════════════════════════════════════════════════════
def fig_training_curves(runs):
    """2x2 grid: total val loss, caption loss, contrastive loss, R@1."""

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Training Dynamics — Validation Metrics per Epoch",
                 fontweight="bold", fontsize=14, y=0.98)

    metrics_spec = [
        ("val_loss",          "Total Validation Loss",   "lower is better", axes[0, 0]),
        ("val_caption",       "Caption Loss",            "lower is better", axes[0, 1]),
        ("val_contrastive",   "Contrastive Loss",        "lower is better", axes[1, 0]),
        ("val_ecg2text_R@1",  "ECG→Text Recall@1",       "higher is better", axes[1, 1]),
    ]

    for metric_col, title, direction, ax in metrics_spec:
        for name in CURVE_EXPERIMENTS:
            if name not in runs:
                continue
            m = runs[name]["metrics"]
            if metric_col not in m.columns:
                continue
            y = m[metric_col]
            if metric_col == "val_ecg2text_R@1":
                y = y * 100  # convert to %

            label_map = {
                "bart": "BART", "biobart": "BioBart", "biogpt": "BioGPT",
                "clinical_t5": "Clin-T5", "gpt2": "GPT-2",
                "bart_proj256": "BART (proj=256)",
                "patchtst_bart": "PatchTST+BART",
            }
            ax.plot(m["epoch"], y,
                    label=label_map.get(name, name),
                    color=CURVE_COLORS.get(name, "#999"),
                    marker="o", markersize=3, linewidth=1.8)

        ax.set_xlabel("Epoch")
        if metric_col == "val_ecg2text_R@1":
            ax.set_ylabel("R@1 (%)")
        else:
            ax.set_ylabel("Loss")
        ax.set_title(f"{title}  ({direction})", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CURVE_EXPERIMENTS),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    path = os.path.join(OUTDIR, "fig_training_curves.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading experiment data...")
    runs = load_runs()
    print(f"Loaded {len(runs)} experiments: {sorted(runs.keys())}")

    print("\n── Figure 1: Ablation Table ──")
    fig_ablation_table(runs)

    print("\n── Figure 2: Retrieval Results ──")
    fig_retrieval(runs)

    print("\n── Figure 3: Generation Examples ──")
    fig_generation(runs)

    print("\n── Figure 4: Training Curves ──")
    fig_training_curves(runs)

    print(f"\nAll figures saved to {OUTDIR}/")
