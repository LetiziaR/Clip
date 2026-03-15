import math
import re
from collections import Counter


_CLINICAL_CONCEPT_PATTERNS = {
    "normal_ecg": [r"\bnormal ecg\b", r"\bnormales ekg\b", r"\bno definite pathology\b"],
    "sinus_rhythm": [r"\bsinus rhythm\b", r"\bsinusrhythmus\b", r"\bsinusrytm\b"],
    "bradycardia": [r"\bbradycardia\b", r"\bsinusbradykardie\b"],
    "tachycardia": [r"\btachycardia\b", r"\bsinustachykardie\b"],
    "atrial_fibrillation": [r"\batrial fibrillation\b", r"\bafib\b", r"\bvorhofflimmern\b"],
    "atrial_flutter": [r"\batrial flutter\b", r"\bvorhofflattern\b"],
    "rbbb": [r"\bright bundle branch block\b", r"\brechtsschenkelblock\b"],
    "lbbb": [r"\bleft bundle branch block\b", r"\blinksschenkelblock\b"],
    "lafb": [r"\bleft anterior fascicular block\b", r"\blinkstyp\b", r"\bleft axis deviation\b"],
    "av_block": [r"\bav block\b", r"\batrioventricular block\b", r"\bav-block\b"],
    "myocardial_infarction": [r"\bmyocardial infarction\b", r"\binfarkt\b", r"\bmyokardschaden\b"],
    "ischemia": [r"\bischemi\w*\b", r"\bischaemi\w*\b", r"\bst depression\b", r"\bst elevation\b"],
    "lvh": [r"\bleft ventricular hypertrophy\b", r"\blvh\b", r"\blinksbelastung\b"],
}


def _safe_word_tokenize(text):
    # Prefer NLTK tokenizer when available; fall back to whitespace split.
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
    from bert_score import score as bertscore_score

    precision, recall, f1 = bertscore_score(
        predictions,
        references,
        model_type=model_type,
        batch_size=batch_size,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False,
    )
    return {
        "bertscore_precision": float(precision.mean().item()),
        "bertscore_recall": float(recall.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
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


def _tokenize(text):
    text = _normalize_text(text)
    if not text:
        return []
    return text.split(" ")


def _ngram_counts(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _bleu_n(pred_tokens, ref_tokens, n):
    pred_counts = _ngram_counts(pred_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)
    if not pred_counts:
        return 0.0

    overlap = 0
    total = 0
    for ngram, count in pred_counts.items():
        overlap += min(count, ref_counts.get(ngram, 0))
        total += count

    precision = overlap / max(total, 1)
    if precision <= 0.0:
        return 0.0

    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    if pred_len == 0:
        return 0.0

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len / max(pred_len, 1)))

    return bp * precision


def _lcs_length(a, b):
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred_tokens, ref_tokens):
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


def compute_text_generation_metrics(
    predictions,
    references,
    full_metrics=False,
    compute_bertscore=False,
    bertscore_model_type="roberta-large",
    bertscore_batch_size=16,
    bertscore_lang="en",
    bertscore_rescale_with_baseline=True,
):
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    if len(predictions) == 0:
        metrics = {
            "count": 0,
            "rougeL_f1": 0.0,
            "avg_pred_tokens": 0.0,
            "avg_ref_tokens": 0.0,
            "clinical_concept_precision": 0.0,
            "clinical_concept_recall": 0.0,
            "clinical_concept_f1": 0.0,
        }
        if full_metrics:
            metrics["exact_match"] = 0.0
            metrics["bleu1"] = 0.0
            metrics["bleu2"] = 0.0
            metrics["meteor"] = 0.0
        if compute_bertscore:
            metrics["bertscore_precision"] = 0.0
            metrics["bertscore_recall"] = 0.0
            metrics["bertscore_f1"] = 0.0
        return metrics

    exact_match = 0
    bleu1_sum = 0.0
    bleu2_sum = 0.0
    rouge_l_sum = 0.0
    pred_len_sum = 0
    ref_len_sum = 0

    for pred_text, ref_text in zip(predictions, references):
        pred_norm = _normalize_text(pred_text)
        ref_norm = _normalize_text(ref_text)

        if full_metrics and pred_norm == ref_norm:
            exact_match += 1

        pred_tokens = _tokenize(pred_norm)
        ref_tokens = _tokenize(ref_norm)

        pred_len_sum += len(pred_tokens)
        ref_len_sum += len(ref_tokens)

        if full_metrics:
            bleu1_sum += _bleu_n(pred_tokens, ref_tokens, n=1)
            bleu2_sum += _bleu_n(pred_tokens, ref_tokens, n=2)
        rouge_l_sum += _rouge_l_f1(pred_tokens, ref_tokens)

    total = len(predictions)
    metrics = {
        "count": total,
        "rougeL_f1": rouge_l_sum / total,
        "avg_pred_tokens": pred_len_sum / total,
        "avg_ref_tokens": ref_len_sum / total,
    }

    if full_metrics:
        metrics["exact_match"] = exact_match / total
        metrics["bleu1"] = bleu1_sum / total
        metrics["bleu2"] = bleu2_sum / total

    metrics.update(_compute_clinical_concept_micro_metrics(predictions, references))

    if full_metrics:
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
