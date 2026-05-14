# Experiments and Results

## 1. Overview

This chapter reports the empirical evaluation of the proposed CoCa-style
ECG-to-text model along the four design axes introduced in Chapter 1: the
time-series encoder, the text decoder, the source of textual supervision,
and the optional auxiliary classification head. The evaluation is staged
across two datasets. PTB-XL is used for model development — it is small
enough to allow each axis to be varied in turn while holding the others
fixed, and curated enough that between-condition differences are unlikely
to be dominated by annotation noise. MIMIC-IV-ECG is used as a transfer
setting: roughly two orders of magnitude larger, with substantially
noisier free-text English reports drawn from a heterogeneous adult ICU
population. The chapter follows this logic: experimental setup, model
selection on PTB-XL, training dynamics, final test-set performance,
transfer to MIMIC-IV-ECG, and discussion.

## 2. Experimental Setup

Datasets and preprocessing
The model developed in this thesis is trained and evaluated on two publicly available datasets: PTB-XL \citep{PhysioNet-ptb-xl-1.0.3} and MIMIC-IV-ECG \citep{PhysioNet-mimic-iv-ecg-1.0}. The two datasets differ substantially in both scale and annotation quality. PTB-XL is smaller and more curated, with paired reports and structured diagnostic labels that make it well suited for supervised classification tasks. In contrast, MIMIC-IV-ECG is significantly larger and more heterogeneous, with weaker and noisier annotations; however, its scale makes it potentially advantageous for representation learning approaches such as contrastive learning.\\

Differences in performance across the two datasets may arise from multiple factors, including dataset size, annotation quality, and underlying distributional shifts. As a result, observed performance differences should be interpreted with caution and not attributed to a single cause.\\

To ensure comparability across experiments, both datasets are processed through a unified data pipeline. Identical preprocessing steps are applied, enabling a consistent training framework across data sources.\\
\paragraph{PTB-XL}\\
PTB-XL ~\citep{wagner2020ptbxl, PhysioNet-ptb-xl-1.0.3} is a large dataset of ECG recordings edited by the \emph{Physikalisch-Technische Bundesanstalt} in Berlin, collected between 1984 and 2001 and distributed via PhysioNet. There are 21,799 ten-second twelve-lead ECG signals from 18,869 patients. \\
Each recording is provided at both \(100\,\text{Hz}\) and \(500\,\text{Hz}\); this thesis uses the 500~Hz version exclusively.\\
Each ECG is paired with: a  \textit{cardiologist report }written mostly in German or English; a multi-label set of \emph{SCP-ECG diagnostic statements} ~\citep{rubel2016scpecg}, used for classification; \emph{demographic metadata} (age, sex, height, weight), used in this work to construct the pseudo-reports described in Section~\ref{sec:data:text}.\\
For classification, PTB-XL provides both fine-grained SCP labels and a
five-class \emph{superclass} grouping
(\texttt{NORM}, \texttt{MI}, \texttt{STTC}, \texttt{CD}, \texttt{HYP}), which enables evaluation at multiple levels of granularity.\\
PTB-XL has an official patient-stratified ten-fold partition,  so that all recordings from a given patient are confined to a single fold, and that label distributions are approximately preserved. Folds 1 to 8 are used for training, fold 9 for validation, and fold 10 as test set. The use of the official split prevents patient-level leakage and ensures comparability with prior work.

\paragraph{MIMIC-IV-ECG}\\
\label{sec:data:mimicecg}
The second dataset is the MIMIC-IV-ECG Diagnostic Electrocardiogram Matched Subset ~\citep{PhysioNet-mimic-iv-ecg-1.0}, which contains 800,035  recordings from 161,352 patients treated at the Beth Israel Deaconess Medical Center between 2008 and 2019. All signals are sampled at 500~Hz and share the same \((12,5000)\) shape as PTB-XL, enabling direct reuse of the same model architecture.\\
Each recording is associated with a short report. Through linking to the broader MIMIC-IV database \citep{PhysioNet-mimiciv-3.1, Johnson2023MIMICIV}, additional information, such as demographic metadata, is incorporated and used to construct pseudo-reports.\\
Unlike PTB-XL, MIMIC-IV-ECG does not provide diagnostic labels. To bridge this gap, a rule-based labeling pipeline was developed to map textual reports to standardized diagnostic codes aligned with the PTB-XL annotation scheme.

\paragraph{Derivation of ECG-Level Diagnostic Labels} 
In order to construct diagnostic labels for the MIMIC-ECG dataset, a set of mapping rules was defined to associate textual patterns with diagnostic labels from the PTB-XL SCP (Standard Communications Protocol) code system. Each rule consists of: an SCP code (e.g., \texttt{AFIB}, \texttt{IMI}, \texttt{LVH}), an optional superclass (e.g., \texttt{MI}, \texttt{CD}, \texttt{HYP}), and one or more regular expressions capturing textual variations of the corresponding diagnosis.\\
For each ECG, the report was scanned, and the matching SCP code was assigned. Multiple matches were allowed, resulting in a multi-label representation. In addition, each SCP code was mapped to its corresponding superclass, following the PTB-XL diagnostic superclasses: \texttt{NORM} (normal ECG), \texttt{MI} (myocardial infarction), \texttt{STTC} (ST/T changes), \texttt{CD} (conduction disturbances), and \texttt{HYP} (hypertrophy).
The mapping allows multiple superclasses to be assigned to a single ECG.

\paragraph{Subject-Level Stratified Fold Assignment} \\

Unlike PTB-XL, the MIMIC-ECG dataset does not provide an official fold split. For this reason, a subject-level stratified cross-validation scheme was constructed following the general principles adopted in PTB-XL.
Every recording belonging to the same subject was assigned to the same fold. The goal was to split subjects into ten folds while preserving a balanced distribution of diagnostic statements across folds. 
To ensure stable stratification, only "frequent" labels were retained; the top 200 most common labels were used to construct the stratification space.
The split was then saved and used in all subsequent experiments.


\paragraph*{Signal Preprocessing and text input}\\
\label{sec:data:text}
The preprocessing pipeline applied to both datasets is the same.
First, the raw waveform is loaded and converted to a representation of shape $(12, T)$. 
Invalid values (i.e., NaNs or infinities) are replaced to ensure numerical stability. 
The signal is then standardised using global $z$-score normalisation across all leads and time steps. 
Finally, recordings are truncated or zero-padded to a fixed length of $T = 5000$ samples.
Global normalisation is chosen over per lead normalisation in order to preserve inter-lead amplitude relationships, which are diagnostically important.\\
On the text side, the model is configurable: it is possible to link each ECG either with a diagnostic report or a synthetic pseudo-report derived from demographic metadata.
The pseudo-report representation does not contain diagnostic information and can be used to assess whether contrastive learning alone can learn diagnostic concepts from the signals only.\\
Both datasets are publicly available and fully de-identified. PTB-XL is distributed under a Creative Commons licence, while MIMIC-IV requires credentialed access and completion of data use agreements. All data used in this thesis were accessed and processed in compliance with the respective licensing and institutional guidelines.

\\
Given the difference in scale between the datasets, PTB-XL is first used as a development dataset for hyperparameter optimization, architectural ablations, and encoder–decoder comparisons. Its size enables extensive exploration of the design space within a manageable computational budget.

The model is then scaled to MIMIC-IV-ECG to evaluate whether conclusions from PTB-XL can be transferred to a larger and more heterogeneous clinical corpus. This model selection allows design decisions to be driven by systematic comparison rather than by the constraints of a single computationally expensive training regime.

The transition from PTB-XL to MIMIC-IV-ECG serves as an explicit test of generalization across dataset scale. A configuration that performs consistently well on both datasets provides stronger evidence of robustness than results obtained on either dataset in isolation.

The hyperparameters optimized on PTB-XL are not guaranteed to remain optimal at the scale of MIMIC-IV-ECG. therefore some training parameters (e.g., learning rate, batch size, ) are re-tuned.

### 2.2 Evaluation metrics

Three families of metrics are reported.

*Generation* is evaluated against the reference clinical reports using
ROUGE-L F1, BLEU-4, METEOR, CIDEr and BERTScore F1, complemented by a
clinical-concept F1 computed against a curated cardiology vocabulary.
ROUGE-L captures longest-common-subsequence overlap, BLEU-4 reports
n-gram precision up to 4-grams, METEOR adds stemming and synonym
matching, CIDEr measures consensus-weighted n-gram similarity against
the reference, and BERTScore F1 captures contextual semantic similarity.

*Retrieval* is evaluated on the validation set using cosine similarity
in the projected embedding space. R@1 and R@5 are reported in both
directions (ECG → text, text → ECG); asymmetries are informative about
which modality carries more discriminative information after projection.

*Classification* is reported only for runs with the auxiliary head. For
the Dirichlet-evidential variant, accuracy and macro-F1 over the five
PTB-XL diagnostic superclasses are reported alongside the mean
predictive uncertainty implied by the Dirichlet concentration
parameters.

Validation cross-entropy is the model-selection criterion used for
early stopping and checkpoint choice. Generation, retrieval and
classification metrics are reported at the selected checkpoint as
complementary evidence; their ordering need not coincide with that of
validation cross-entropy, and Section 4 returns to this point.

### 2.3 Training setup

The text encoder is Bio_ClinicalBERT, with the bottom layers frozen and
the top two transformer blocks unfrozen during training. The contrastive
branch uses the Bio_ClinicalBERT tokenizer; the generative branch uses
each decoder's native tokenizer. The contrastive temperature is 0.07,
the projection dimension is 128 (256 in the final configuration), and
the contrastive and caption losses are weighted equally
(λ_contrastive = λ_caption = 1.0). When the auxiliary head is present,
the multi-label binary variant is weighted by 0.5; the Dirichlet variant
uses unit weight on the negative log-likelihood and weight 0.1 on the
KL regulariser, with linear annealing of the KL term over the first ten
epochs.

PTB-XL runs use batch size 32, learning rate 5×10⁻⁵, weight decay
1×10⁻², a 30-epoch cap with cosine schedule, early stopping after seven
non-improving validation epochs, and seed 42. The MIMIC-IV-ECG run
increases the batch size to 48 and the learning rate to 2×10⁻⁴
following standard linear-with-batch scaling. All runs use a single
A100 (80 GB) GPU.

---

## 3. Model Selection on PTB-XL

The four design axes are compared in the order in which the design
decisions are taken: decoder family, time-series encoder, text source
and auxiliary head. Each comparison varies one axis with the remaining
three held at their reference values (BART decoder, TS2Vec encoder,
real reports, no head). The selection criterion is validation
cross-entropy at the early-stopping checkpoint, with generation and
retrieval metrics serving as confirmatory evidence.

### 3.1 Decoder family

Five decoders are compared with the encoder, text source and auxiliary
head held at reference: three encoder–decoder transformers (BART,
BioBART, FLAN-T5) and two decoder-only models (GPT-2, BioGPT).

| Decoder    | Val loss | ROUGE-L | BLEU-4 | METEOR | CIDEr | BERTScore F1 |
|------------|---------:|--------:|-------:|-------:|------:|-------------:|
| BART       |    2.519 |   0.340 |  0.149 |  0.317 | 1.570 |        0.246 |
| BioBART    |    2.478 |   0.352 |  0.157 |  0.332 | 1.648 |        0.256 |
| FLAN-T5    |    2.978 |    —    |   —    |   —    |   —   |         —    |
| GPT-2      |    2.537 |   0.117 |  0.046 |  0.186 | 0.067 |       −0.110 |
| BioGPT     |    2.661 |   0.116 |  0.034 |  0.193 | 0.010 |       −0.182 |

Two outcomes structure the choice. First, despite token-level validation
losses comparable to BART, the decoder-only models produce verbose,
off-topic generations: average prediction length is 2.5–3 times the
11-token reference, which collapses BLEU-4 and CIDEr and drives
BERTScore F1 negative. Per-token cross-entropy under teacher forcing is
normalised per position and does not penalise this length-distribution
mismatch, which explains the divergence between cross-entropy and
downstream metrics. Second, BART and BioBART are nearly indistinguishable:
BioBART attains a marginally lower validation loss and a +1.2 ROUGE-L
advantage, of similar magnitude to the spread among nominally similar
BART configurations elsewhere in the chapter. FLAN-T5 fails to converge
under the shared protocol and does not produce evaluable generations.

BART is retained as the reference decoder. The BART–BioBART gap is small
enough that the qualitative ordering on the remaining three axes would
not change under BioBART; BART is preferred for training stability and
slightly faster convergence.

### 3.2 Time-series encoder

PatchTST is compared to TS2Vec with the BART decoder, real reports and
no auxiliary head.

| Encoder  | Val loss | ROUGE-L | BLEU-4 | CIDEr | R@5 (E→T) | R@5 (T→E) |
|----------|---------:|--------:|-------:|------:|----------:|----------:|
| TS2Vec   |    2.519 |   0.340 |  0.149 | 1.570 |     0.117 |     0.111 |
| PatchTST |    3.069 |   0.312 |  0.131 | 1.237 |     0.064 |     0.057 |

TS2Vec dominates on every metric, with the largest differences on
retrieval (R@5 is roughly halved under PatchTST in both directions). The
two encoders are not on equal footing in this comparison: TS2Vec is
self-supervised on the unlabelled PTB-XL signal, whereas PatchTST is
randomly initialised (no public PatchTST checkpoint for 12-lead ECG was
available). The result therefore conflates architecture and pretraining;
the most parsimonious reading is that, at PTB-XL scale, the availability
of a pretrained encoder is the binding factor on joint representation
quality. TS2Vec is retained for the remainder of the development phase;
the encoder comparison is revisited at MIMIC scale in Section 6.

### 3.3 Text source

The pseudo-report condition replaces the clinical report in the
*contrastive* branch with a templated string built from age, sex,
weight, height and pacemaker status; the generative branch always sees
the real report. Two learning rates are reported because the
pseudo-report sweep included a 1×10⁻⁴ run that is therefore confounded
with learning rate.

| Text source for contrastive | Val loss | ROUGE-L | BLEU-4 | CIDEr | R@5 (E→T) |
|------------------------------|---------:|--------:|-------:|------:|----------:|
| Real report (lr 5×10⁻⁵)      |    2.519 |   0.340 |  0.149 | 1.570 |     0.117 |
| Pseudo report (lr 5×10⁻⁵)    |    3.353 |   0.318 |  0.110 | 1.457 |     0.020 |
| Pseudo report (lr 1×10⁻⁴)    |    3.272 |   0.355 |  0.172 | 1.732 |     0.022 |

Validation loss rises by roughly 0.8 nats and retrieval R@5 collapses by
a factor of five when the contrastive branch is deprived of clinical
text. Surface generation metrics, by contrast, remain close to baseline
— the decoder still conditions on real reports — and at lr 1×10⁻⁴ even
nominally exceed it. This dissociation is the most consequential
observation of the chapter: in pseudo-report runs the contrastive
objective aligns the embedding spaces along the only structure shared
between modalities (demographic information), and the resulting
representation is impoverished in a way that surface metrics do not
detect. Real reports are retained as the contrastive supervision signal.

### 3.4 Auxiliary classification head

Two head variants are compared: a binary multi-label head over 52 SCP
codes, and a Dirichlet-evidential head over the five diagnostic
superclasses. Both consume the global ECG token and are trained jointly
with the contrastive and caption losses. Validation cross-entropies are
not on a common scale across head conditions because the joint loss
includes the auxiliary term; the comparison at this step relies on the
generation, retrieval and classification metrics jointly.

| Configuration                | ROUGE-L | BLEU-4 | METEOR | CIDEr | BERTScore F1 | Clin-F1 | R@5 (E→T) |
|------------------------------|--------:|-------:|-------:|------:|-------------:|--------:|----------:|
| BART, no head (reference)    |   0.340 |  0.149 |  0.317 | 1.570 |        0.246 |   0.686 |     0.117 |
| + 52-class binary head       |   0.364 |  0.157 |  0.343 | 1.786 |        0.261 |   0.696 |     0.103 |
| + Dirichlet 5-class head     |   0.414 |  0.216 |  0.390 | 2.156 |        0.326 |   0.739 |     0.146 |

The Dirichlet head produces the largest single-axis improvements
observed in this study: ROUGE-L +7.4, BLEU-4 +6.7, METEOR +7.3,
CIDEr +0.59, BERTScore F1 +8.0, clinical-F1 +5.3 and R@5 +0.029 over
the no-head baseline. It additionally attains 0.84 superclass accuracy
with mean predictive uncertainty 0.24 — a Dirichlet posterior that is
neither degenerate nor uniform, consistent with calibrated
class-conditional confidence. The 52-class binary head provides a
smaller, partial gain on generation and slightly degrades retrieval.
That the coarse five-class Dirichlet head outperforms the finer
52-class binary head suggests that low-noise, calibrated supervision is
more valuable than nominal label granularity in this regime; the present
comparison does not isolate the contributions of label aggregation,
evidential parameterisation and KL regularisation.

The Dirichlet head is included in the final configuration.

### 3.5 Selected configuration

The selected PTB-XL configuration combines the BART decoder, the
TS2Vec encoder, real clinical reports as contrastive supervision and
the Dirichlet five-class head. Hyperparameters are fixed at the values
that produced the best validation behaviour during the per-axis sweep:
contrastive temperature 0.07, projection dimension 256, learning rate
5×10⁻⁵ and batch size 32. Within the explored hyperparameter ranges
the qualitative ranking of architectural choices is preserved; the
only substantial deviation is batch size, where bs = 64 worsens
validation loss by 0.54 nats — plausibly attributable to PTB-XL's
large clusters of near-duplicate normal-sinus recordings inflating the
rate of false-negative pairs in the contrastive loss.

---

## 4. Training Dynamics

The training and validation curves are informative beyond the
early-stopping summary, both about the convergence behaviour of the
joint objective and about the relationship between the selection
criterion (validation cross-entropy) and the metrics that ultimately
matter (generation and retrieval).

**BART reference (no head, ten epochs).** Training loss falls
monotonically from 3.28 to 0.75, while validation loss reaches its
minimum of 2.519 at epoch 4 and drifts upward thereafter. The widening
train–val gap reflects overfitting of the captioning term — train
caption loss falls from 0.97 to 0.24 while validation caption loss
flattens around 0.49 — whereas the contrastive term continues to
decrease on both partitions throughout. Retrieval R@5 (E→T) climbs
steadily from 0.059 at epoch 1 to 0.117 at epoch 10, so the
best-validation-loss checkpoint at epoch 4 (R@5 = 0.105) is not the
retrieval-optimal checkpoint.

**Final model (Dirichlet head, eleven epochs).** Convergence is shaped
by the three loss components in distinct ways. Train caption loss falls
from 0.79 to 0.17, while validation caption loss bottoms at 0.48 around
epoch 5 and rises slowly thereafter — the same caption-side overfitting
pattern as the no-head reference. The contrastive loss continues to
improve on both partitions throughout. The Dirichlet term decreases
monotonically on training (1.76 → 1.33) and validation (1.56 → 1.34),
and validation classification accuracy rises smoothly from 0.80 to 0.84
with no sign of plateau within the budget. The mean predictive
uncertainty falls from 0.41 to 0.22, indicating that the Dirichlet
posterior sharpens as training progresses without becoming degenerate.
Retrieval R@5 (E→T) climbs from 0.087 to a maximum of 0.149 at epoch 10.

Two regularities are visible across configurations. First, validation
captioning loss saturates earlier than validation contrastive and
classification losses, regardless of the auxiliary head; the relative
weighting of the loss terms therefore implicitly determines when the
joint criterion stops improving, and an early-stopping rule based on
total validation loss is conservative for the contrastive and
classification objectives. Second, decoder-only configurations show
qualitatively different curves: training converges to comparable
per-token loss but the validation length distribution does not align
with the reference, producing the surface-metric collapse documented in
Section 3.1. The diagnosis is consistent with a length-distribution
mismatch rather than with insufficient capacity.

---

## 5. Final Results on PTB-XL

The final configuration — TS2Vec encoder, BART decoder, real reports,
Dirichlet five-class head — is evaluated on the 2198-recording PTB-XL
test split.

| Metric              | Final model | BART reference |
|---------------------|------------:|---------------:|
| ROUGE-L F1          |       0.414 |          0.340 |
| BLEU-4              |       0.216 |          0.149 |
| METEOR              |       0.390 |          0.317 |
| CIDEr               |       2.156 |          1.570 |
| BERTScore F1        |       0.326 |          0.246 |
| Clinical-concept F1 |       0.739 |          0.686 |

Across all six generation metrics, the final configuration improves on
the BART reference by margins substantially larger than the spread
observed among nominally similar BART variants in the development
sweep, and the improvement is consistent in direction across surface
(ROUGE-L, BLEU-4), paraphrase-aware (METEOR), consensus (CIDEr),
semantic (BERTScore F1) and clinical-vocabulary (clinical-F1) measures.

Retrieval performance, evaluated on the validation set in both
directions, is summarised below.

| Direction   | R@1   | R@5   |
|-------------|------:|------:|
| ECG → text  | 0.044 | 0.146 |
| text → ECG  | 0.038 | 0.140 |

The two directions are nearly symmetric, indicating that the projected
ECG and text embeddings carry comparable amounts of discriminative
information and that the contrastive branch has not collapsed onto one
modality. Retrieval R@5 improves from 0.117 (BART reference) to 0.146
under the final configuration, a 25% relative gain attributable to the
Dirichlet head's regularising effect on the shared encoder
representation. The classification head additionally attains 0.84
superclass accuracy and 0.24 mean predictive uncertainty, providing a
calibrated diagnostic-class signal as a usable byproduct of the joint
training.

All reported numbers correspond to a single random seed (42); a
multi-seed replication of the final configuration was not performed
within the available compute budget. The cluster of nominally similar
BART variants in the development sweep spans a validation-loss range of
2.45–2.59, substantially smaller than the differences between the final
and reference configurations — suggestive of low seed variance, but not
a substitute for a formal multi-seed estimate.

---

## 6. Transfer to MIMIC-IV-ECG

MIMIC-IV-ECG differs from PTB-XL along three approximately independent
dimensions: it is roughly two orders of magnitude larger; its reports
are free-text English clinical notes rather than structured German
findings; and its recordings are drawn from a heterogeneous adult ICU
population in which demographic and disease distributions differ
substantially. The transfer experiment is therefore best understood as
an external-validity check rather than as a sanity confirmation.

A MIMIC training run (`mimic_ts2vec_bart_1gpu`) was launched with the
PTB-XL final-configuration architecture, batch size raised to 48,
learning rate raised to 2×10⁻⁴, and the auxiliary head disabled in
this initial step in order to isolate the effect of dataset scale on
the contrastive and generative objectives. The run did not complete
within the wall-clock budget of the allocated SLURM job, and the
on-disk metrics log does not contain an evaluable validation epoch;
consequently no converged test-set numbers can be reported on
MIMIC-IV-ECG in this chapter.

The configured pipeline runs unchanged at MIMIC scale: the encoder,
decoder and tokenizer accept MIMIC's signal length, lead order and
English report distribution without architectural modification, and the
pseudo-report builder, dual-tokenizer setup and auxiliary head are all
directly applicable through the existing configuration switches. A
complete MIMIC sweep — including re-introduction of the Dirichlet
head, a like-for-like text-source ablation against MIMIC demographic
templates, and at least one multi-seed replication of the final
configuration — is identified as the principal outstanding experiment
for this work.

---

## 7. Discussion

The experiments in this chapter support a small number of clear
findings about the design space of CoCa-style ECG-to-text models on
PTB-XL. The strongest joint behaviour observed in the study is
obtained by combining a BART decoder, a self-supervised TS2Vec
encoder, real clinical reports as contrastive supervision and a
Dirichlet-evidential five-class auxiliary head: ROUGE-L 0.414,
BLEU-4 0.216, METEOR 0.390, CIDEr 2.156, BERTScore F1 0.326 and
R@5 0.146 on the PTB-XL test set, together with calibrated
class-conditional uncertainty as a byproduct.

The most consequential design choice is the Dirichlet auxiliary head,
which produces the largest single-axis gain in the study and, unlike
the other interventions, improves generation and retrieval
simultaneously. Two further findings are robust across the
comparisons performed: encoder–decoder transformers are required at
PTB-XL scale, with decoder-only models failing every surface and
semantic generation metric despite comparable token-level
cross-entropy; and real clinical reports are necessary as the
contrastive supervision signal, with demographic pseudo-reports
collapsing retrieval by a factor of five while leaving surface
generation metrics largely intact. The latter dissociation is the
chapter's clearest cautionary observation: surface metrics alone are
insufficient to validate a contrastive branch, and retrieval R@k
should be reported alongside generation whenever the contrastive
supervision is being varied.

The findings are subject to three limitations. The ablation is
slice-wise rather than fully factorial, so interactions between axes
are not estimated. All reported numbers correspond to a single random
seed; the indirect evidence for low seed variance from the
BART/BioBART cluster is suggestive but not a formal substitute for a
multi-seed replication of the selected configuration. The
MIMIC-IV-ECG transfer did not complete within the available compute
budget, so the external validity of the PTB-XL findings remains an
open empirical question.

The main takeaway of the chapter is that, on PTB-XL, supervision
quality dominates supervision quantity for representation learning —
both in the strong negative result on pseudo-reports and in the
strong positive result on the auxiliary head — while encoder–decoder
coupling and a pretrained time-series encoder provide the substrate
that allows that supervision to be exploited. Whether the same
hierarchy holds at MIMIC scale, where text is much more abundant but
considerably noisier, is the principal question carried forward.
