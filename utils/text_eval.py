"""Text-generation metrics for ECG report generation.

BLEU-4, ROUGE-L and BERTScore go through TorchMetrics so the numerical
implementations are the same ones used across the PyTorch ecosystem
(and shared with PyTorch Lightning workflows).  CIDEr-D and METEOR are
not in TorchMetrics, so they keep their canonical reference backends:

    BLEU-4    - torchmetrics.text.SacreBLEUScore
    ROUGE-L   - torchmetrics.text.ROUGEScore       (Porter stemmer)
    BERTScore - torchmetrics.text.BERTScore        (contextual embeddings)
    CIDEr-D   - pycocoevalcap.cider.cider.Cider    (MS-COCO caption eval)
    METEOR    - nltk.translate.meteor_score        (WordNet synonyms)

Clinical-concept P/R/F1 is a task-specific metric computed from regex
patterns grounded in the PTB-XL SCP taxonomy.
"""

import re


# Clinical concept patterns grounded in the PTB-XL SCP taxonomy
# (scp_statements.csv).  Each key documents the SCP codes it covers.
# Patterns include English, German, and Swedish variants observed in
# PTB-XL reports.
_CLINICAL_CONCEPT_PATTERNS = {
    # ── NORM superclass ──
    # SCP: NORM
    "normal_ecg": [
        r"\bnormal ecg\b", r"\bnormal ekg\b", r"\bnormales ekg\b",
        r"\bno definite pathology\b",
    ],
    # ── Rhythm (SCP rhythm codes) ──
    # SCP: SR
    "sinus_rhythm": [
        r"\bsinus rhythm\b", r"\bsinusrhythmus\b", r"\bsinusrytm\b",
    ],
    # SCP: SBRAD
    "sinus_bradycardia": [
        r"\bsinus bradycardia\b", r"\bbradycardia\b",
        r"\bsinusbradykardie\b", r"\bbradykard\w*\b",
    ],
    # SCP: STACH
    "sinus_tachycardia": [
        r"\bsinus tachycardia\b", r"\btachycardia\b",
        r"\bsinustachykardie\b", r"\btachykard\w*\b",
    ],
    # SCP: SARRH
    "sinus_arrhythmia": [
        r"\bsinus arrhythmia\b", r"\bsinusarrhythmie\b",
        r"\bsinus arrhythmie\b", r"\bsinus arytmi\b",
    ],
    # SCP: AFIB
    "atrial_fibrillation": [
        r"\batrial fibrillation\b", r"\bafib\b",
        r"\bvorhofflimmern\b", r"\bf[oö]rmaksflimmer\b",
    ],
    # SCP: AFLT
    "atrial_flutter": [
        r"\batrial flutter\b", r"\bvorhofflattern\b",
    ],
    # SCP: SVARR, SVTAC, PSVT
    "supraventricular_arrhythmia": [
        r"\bsupraventricular tachycardia\b", r"\bsvt\b",
        r"\bsupraventricular arrhythmia\b",
        r"\bsupraventr\w*\s*tachykard\w*\b",
        r"\bparoxysmal supraventricular\b",
    ],
    # SCP: PACE
    "pacemaker": [
        r"\bpacemaker\b", r"\bschrittmacher\b",
    ],
    # SCP: PVC
    "premature_ventricular_complex": [
        r"\bventricular premature\b", r"\bpremature ventricular\b",
        r"\bventrikul[aä]re? extrasystol\w*\b", r"\bpvc\b",
    ],
    # SCP: PAC, PRC(S)
    "premature_atrial_complex": [
        r"\batrial premature\b", r"\bpremature atrial\b",
        r"\bsupraventr\w*\s*extrasystol\w*\b", r"\bpac\b",
        r"\bpremature complex\b",
    ],
    # ── CD superclass (conduction disturbances) ──
    # SCP: CRBBB, IRBBB
    "rbbb": [
        r"\bright bundle branch block\b", r"\brbbb\b",
        r"\brechtsschenkelblock\b",
        r"\bincomplete right bundle branch\b",
    ],
    # SCP: CLBBB, ILBBB
    "lbbb": [
        r"\bleft bundle branch block\b", r"\blbbb\b",
        r"\blinksschenkelblock\b",
        r"\bincomplete left bundle branch\b",
    ],
    # SCP: LAFB, LPFB
    "fascicular_block": [
        r"\bleft anterior fascicular block\b", r"\blafb\b",
        r"\bleft posterior fascicular block\b", r"\blpfb\b",
        r"\blinkstyp\b", r"\bleft axis deviation\b",
        r"\bueberdrehter linkstyp\b",
        r"\bhemiblock\b", r"\blinksposteriorer hemiblock\b",
    ],
    # SCP: 1AVB, 2AVB, 3AVB
    "av_block": [
        r"\bav[- ]?block\b", r"\batrioventricular block\b",
        r"\bfirst degree av\b", r"\bsecond degree av\b",
        r"\bthird degree av\b",
    ],
    # SCP: IVCD
    "ivcd": [
        r"\bintraventricular conduction\b",
        r"\bivcd\b", r"\bintraventrikularer block\b",
    ],
    # SCP: WPW
    "wpw": [
        r"\bwolf.?parkinson.?white\b", r"\bwpw\b",
    ],
    # ── MI superclass (myocardial infarction) ──
    # SCP: IMI, ASMI, AMI, ALMI, ILMI, LMI, IPLMI, IPMI, PMI,
    #       INJAS, INJAL, INJIN, INJLA, INJIL
    "myocardial_infarction": [
        r"\bmyocardial infarction\b", r"\binfarkt\b",
        r"\bmyokardschaden\b", r"\binfarction\b",
        r"\bsubendocardial injury\b",
    ],
    # ── STTC superclass (ST/T changes) ──
    # SCP: ISC_, ISCAL, ISCIN, ISCIL, ISCAS, ISCLA, ISCAN
    "ischemia": [
        r"\bischemi\w*\b", r"\bischaemi\w*\b",
        r"\bisch[aä]mie\b",
    ],
    # SCP: STD_, STE_, NST_
    "st_changes": [
        r"\bst[- ]?depression\b", r"\bst[- ]?elevation\b",
        r"\bst[- ]?segment\w*\b", r"\bst[- ]?change\w*\b",
        r"\bst & t abnorm\b", r"\bst abnorm\b",
        r"\bst[- ]?senkung\b", r"\bst[- ]?hebung\b",
    ],
    # SCP: NDT, NT_, INVT, LOWT, TAB_
    "t_wave_changes": [
        r"\bt[- ]?wave\w*\b", r"\bt[- ]?abnorm\w*\b",
        r"\binverted t\b", r"\bt[- ]?inversion\b",
        r"\bflattened t\b", r"\blow.?amplitude t\b",
        r"\bt-negativierung\b", r"\bt-ver[aä]nderung\w*\b",
    ],
    # SCP: LNGQT
    "long_qt": [
        r"\blong qt\b", r"\bprolonged qt\b",
        r"\bqt[- ]?verl[aä]ngerung\b",
    ],
    # ── HYP superclass (hypertrophy) ──
    # SCP: LVH, VCLVH
    "lvh": [
        r"\bleft ventricular hypertrophy\b", r"\blvh\b",
        r"\blinkshypertrophie\b", r"\blinksbelastung\b",
        r"\bvoltage criteria.{0,20}hypertrophy\b",
        r"\bkammarhypertrofi\b",
    ],
    # SCP: RVH
    "rvh": [
        r"\bright ventricular hypertrophy\b", r"\brvh\b",
        r"\brechtshypertrophie\b", r"\brechtsbelastung\b",
    ],
    # SCP: LAO/LAE, RAO/RAE
    "atrial_enlargement": [
        r"\batrial overload\b", r"\batrial enlargement\b",
        r"\bp[- ]?sinistrocardiale\b", r"\bp[- ]?mitrale\b",
        r"\bp[- ]?pulmonale\b", r"\bp[- ]?dextrocardiale\b",
        r"\bvorhofbelastung\b",
    ],
    # SCP: QWAVE
    "q_waves": [
        r"\bq[- ]?wave\w*\b", r"\bq[- ]?zack\w*\b",
        r"\bpathologische q\b",
    ],
    # SCP: LVOLT, HVOLT
    "voltage_abnormality": [
        r"\blow.{0,5}voltage\w*\b", r"\bhigh.{0,5}voltage\w*\b",
        r"\bniedervoltage\b",
    ],
}


def _normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_clinical_concepts(text):
    normalized = _normalize_text(text)
    found = set()
    for concept, patterns in _CLINICAL_CONCEPT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                found.add(concept)
                break
    return found


def _compute_clinical_concept_micro_metrics(predictions, references):
    tp = 0
    fp = 0
    fn = 0

    for pred_text, ref_text in zip(predictions, references):
        pred_set = _extract_clinical_concepts(pred_text)
        ref_set = _extract_clinical_concepts(ref_text)
        tp += len(pred_set & ref_set)
        fp += len(pred_set - ref_set)
        fn += len(ref_set - pred_set)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denom = precision + recall
    f1 = 0.0 if denom == 0 else (2 * precision * recall / denom)
    return {
        "clinical_concept_precision": precision,
        "clinical_concept_recall": recall,
        "clinical_concept_f1": f1,
    }


def _to_scalar(value):
    """Convert a torchmetrics output (tensor / list / scalar) to a float."""
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _compute_bleu4(predictions, references):
    """Corpus-level BLEU-4 via torchmetrics.SacreBLEUScore (0-1 scale)."""
    from torchmetrics.text import SacreBLEUScore

    bleu = SacreBLEUScore(n_gram=4)
    score = bleu(predictions, [[ref] for ref in references])
    return _to_scalar(score)


def _compute_rouge_l(predictions, references):
    """Sentence-averaged ROUGE-L F1 via torchmetrics.ROUGEScore (Porter stem)."""
    from torchmetrics.text import ROUGEScore

    rouge = ROUGEScore(rouge_keys=("rougeL",), use_stemmer=True)
    result = rouge(predictions, references)
    return _to_scalar(result["rougeL_fmeasure"])


def _compute_cider_d(predictions, references):
    """CIDEr-D via pycocoevalcap (TF-IDF n-gram cosine, Gaussian length penalty)."""
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [_normalize_text(r)] for i, r in enumerate(references)}
    res = {i: [_normalize_text(p)] for i, p in enumerate(predictions)}
    cider = Cider()
    score, _ = cider.compute_score(gts, res)
    return float(score)


def _safe_word_tokenize(text):
    try:
        from nltk.tokenize import word_tokenize

        return word_tokenize(text)
    except Exception:
        return text.split()


def _compute_meteor(predictions, references):
    from nltk.translate.meteor_score import single_meteor_score

    total = 0.0
    for pred_text, ref_text in zip(predictions, references):
        ref_tokens = _safe_word_tokenize(_normalize_text(ref_text))
        pred_tokens = _safe_word_tokenize(_normalize_text(pred_text))
        total += single_meteor_score(ref_tokens, pred_tokens)
    return total / max(len(predictions), 1)


def _compute_bertscore(
    predictions,
    references,
    model_type="roberta-large",
    batch_size=16,
    lang="en",
    rescale_with_baseline=True,
):
    """BERTScore via torchmetrics.text.BERTScore."""
    from torchmetrics.text.bert import BERTScore

    bertscore = BERTScore(
        model_name_or_path=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        batch_size=batch_size,
    )
    result = bertscore(predictions, references)
    return {
        "bertscore_precision": _to_scalar(result["precision"]),
        "bertscore_recall": _to_scalar(result["recall"]),
        "bertscore_f1": _to_scalar(result["f1"]),
    }


def compute_text_generation_metrics(
    predictions,
    references,
    full_metrics=False,
    compute_bertscore=False,
    compute_clinical_concepts=False,
    bertscore_model_type="roberta-large",
    bertscore_batch_size=16,
    bertscore_lang="en",
    bertscore_rescale_with_baseline=True,
):
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    total = len(predictions)
    if total == 0:
        metrics = {
            "count": 0,
            "rougeL_f1": 0.0,
            "avg_pred_tokens": 0.0,
            "avg_ref_tokens": 0.0,
        }
        if compute_clinical_concepts:
            metrics["clinical_concept_precision"] = 0.0
            metrics["clinical_concept_recall"] = 0.0
            metrics["clinical_concept_f1"] = 0.0
        if full_metrics:
            metrics["bleu4"] = 0.0
            metrics["cider_d"] = 0.0
            metrics["meteor"] = 0.0
        if compute_bertscore:
            metrics["bertscore_precision"] = 0.0
            metrics["bertscore_recall"] = 0.0
            metrics["bertscore_f1"] = 0.0
        return metrics

    pred_len_sum = 0
    ref_len_sum = 0
    for pred_text, ref_text in zip(predictions, references):
        pred_len_sum += len(_normalize_text(pred_text).split(" "))
        ref_len_sum += len(_normalize_text(ref_text).split(" "))

    metrics = {
        "count": total,
        "rougeL_f1": _compute_rouge_l(predictions, references),
        "avg_pred_tokens": pred_len_sum / total,
        "avg_ref_tokens": ref_len_sum / total,
    }

    if compute_clinical_concepts:
        metrics.update(_compute_clinical_concept_micro_metrics(predictions, references))

    if full_metrics:
        metrics["bleu4"] = _compute_bleu4(predictions, references)
        metrics["cider_d"] = _compute_cider_d(predictions, references)
        try:
            metrics["meteor"] = _compute_meteor(predictions, references)
        except Exception as exc:
            metrics["meteor_error"] = str(exc)

    if compute_bertscore:
        try:
            metrics.update(
                _compute_bertscore(
                    predictions,
                    references,
                    model_type=bertscore_model_type,
                    batch_size=bertscore_batch_size,
                    lang=bertscore_lang,
                    rescale_with_baseline=bertscore_rescale_with_baseline,
                )
            )
        except Exception as exc:
            metrics["bertscore_error"] = str(exc)

    return metrics
