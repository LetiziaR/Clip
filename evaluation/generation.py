import torch

from utils.text_eval import compute_text_generation_metrics


def _build_decoder_ecg_tokens(model_ref, x_ts, batch, device):
    """Mirror CoCa.forward's ECG-token path so generation matches training.

    Why: training optionally routes tokens through a Perceiver-IO bottleneck
    and/or prepends Dirichlet disease-context tokens. Skipping either step at
    inference creates a train/eval regime mismatch.
    """
    ts_tokens = model_ref.ts_enc(x_ts)

    if getattr(model_ref, "use_perceiver", False):
        latents = model_ref.perceiver(ts_tokens)
        mode = getattr(model_ref, "perceiver_mode", "both")
        if mode == "both":
            ts_global = latents[:, 0]
            ts_temporal = latents[:, 1:]
        elif mode == "global_only":
            ts_global = latents[:, 0]
            ts_temporal = ts_tokens[:, 1:]
        else:  # "decoder_only"
            ts_global = ts_tokens[:, 0]
            ts_temporal = latents
    else:
        ts_global = ts_tokens[:, 0]
        ts_temporal = ts_tokens[:, 1:]

    if not getattr(model_ref, "use_dirichlet", False):
        return ts_temporal

    if getattr(model_ref, "dirichlet_use_text", False):
        if not isinstance(batch, dict) or "input_ids" not in batch:
            raise ValueError(
                "dirichlet_use_text=True requires batch['input_ids'] at eval time"
            )
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        lang_out = model_ref.language_enc(input_ids=input_ids, attention_mask=attn_mask)
        text_cls = lang_out[0] if isinstance(lang_out, tuple) else lang_out
        dirichlet_input = torch.cat([ts_global, text_cls], dim=-1)
    else:
        dirichlet_input = ts_global

    _, disease_probs, uncertainty = model_ref.dirichlet_head(dirichlet_input)

    if getattr(model_ref, "use_uncertainty", True):
        disease_tokens = model_ref.disease_conditioner(disease_probs, uncertainty)
    else:
        disease_tokens = model_ref.disease_conditioner(disease_probs)

    if getattr(model_ref, "disable_disease_tokens", False):
        disease_tokens = torch.zeros_like(disease_tokens)

    return torch.cat([disease_tokens, ts_temporal], dim=1)


def evaluate_generation(
    model,
    data_loader,
    generation_tokenizer,
    reference_tokenizer,
    max_new_tokens,
    num_beams,
    do_sample,
    temperature,
    top_p,
    no_repeat_ngram_size,
    repetition_penalty,
    length_penalty,
    max_batches,
    full_metrics,
    compute_bertscore,
    bertscore_model_type,
    bertscore_model_alias,
    bertscore_batch_size,
    bertscore_lang,
    bertscore_rescale_with_baseline,
    compute_clinical_concepts=False,
):
    model_ref = model.module if hasattr(model, "module") else model
    model_ref.eval()
    device = next(model_ref.parameters()).device
    decoder_model = getattr(getattr(model_ref, "decoder", None), "model", None)

    if decoder_model is not None:
        cfg = getattr(decoder_model, "config", None)
        model_type = getattr(cfg, "model_type", "") if cfg is not None else ""

    predictions = []
    references = []

    bos_token_id = getattr(generation_tokenizer, "bos_token_id", None)
    eos_token_id = getattr(generation_tokenizer, "eos_token_id", None)
    pad_token_id = getattr(generation_tokenizer, "pad_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(generation_tokenizer, "cls_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(generation_tokenizer, "sep_token_id", None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                x_ts = batch["ecg"].to(device)
                ref_ids = batch.get("decoder_input_ids", batch["input_ids"])
            else:
                if len(batch) < 2:
                    raise ValueError("Batch must contain at least (x_ts, input_ids)")
                x_ts, ref_ids = batch[:2]
                x_ts = x_ts.to(device)

            decoder_ecg_tokens = _build_decoder_ecg_tokens(model_ref, x_ts, batch, device)

            generated_ids = model_ref.decoder.generate(
                ecg_tokens=decoder_ecg_tokens,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            generated_ids = generated_ids.detach().cpu()

            pred_texts = generation_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            raw_pred_texts = generation_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            ref_texts = reference_tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

            normalized_preds = []
            for pred_text, raw_text in zip(pred_texts, raw_pred_texts):
                pred_norm = pred_text.strip()
                if not pred_norm:
                    fallback = raw_text.replace("<pad>", "").replace("</s>", "").replace("<s>", "").strip()
                    pred_norm = fallback
                normalized_preds.append(pred_norm)

            predictions.extend(normalized_preds)
            references.extend([t.strip() for t in ref_texts])

    metrics = compute_text_generation_metrics(
        predictions,
        references,
        full_metrics=full_metrics,
        compute_bertscore=compute_bertscore,
        compute_clinical_concepts=compute_clinical_concepts,
        bertscore_model_type=bertscore_model_type,
        bertscore_model_alias=bertscore_model_alias,
        bertscore_batch_size=bertscore_batch_size,
        bertscore_lang=bertscore_lang,
        bertscore_rescale_with_baseline=bertscore_rescale_with_baseline,
    )
    return metrics, predictions, references
