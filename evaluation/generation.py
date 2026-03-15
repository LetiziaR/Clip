import torch

from utils.text_eval import compute_text_generation_metrics


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
    max_batches,
    full_metrics,
    compute_bertscore,
    bertscore_model_type,
    bertscore_batch_size,
    bertscore_lang,
    bertscore_rescale_with_baseline,
):
    model_ref = model.module if hasattr(model, "module") else model
    model_ref.eval()
    device = next(model_ref.parameters()).device

    predictions = []
    references = []

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

            ts_tokens = model_ref.ts_enc(x_ts)

            bos_token_id = getattr(generation_tokenizer, "bos_token_id", None)
            eos_token_id = getattr(generation_tokenizer, "eos_token_id", None)
            pad_token_id = getattr(generation_tokenizer, "pad_token_id", None)
            if bos_token_id is None:
                bos_token_id = getattr(generation_tokenizer, "cls_token_id", None)
            if eos_token_id is None:
                eos_token_id = getattr(generation_tokenizer, "sep_token_id", None)

            generated_ids = model_ref.decoder.generate(
                ecg_tokens=ts_tokens,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            generated_ids = generated_ids.detach().cpu()
            ref_ids = ref_ids.detach().cpu() if hasattr(ref_ids, "detach") else ref_ids

            pred_texts = generation_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            ref_texts = reference_tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

            predictions.extend([t.strip() for t in pred_texts])
            references.extend([t.strip() for t in ref_texts])

    metrics = compute_text_generation_metrics(
        predictions,
        references,
        full_metrics=full_metrics,
        compute_bertscore=compute_bertscore,
        bertscore_model_type=bertscore_model_type,
        bertscore_batch_size=bertscore_batch_size,
        bertscore_lang=bertscore_lang,
        bertscore_rescale_with_baseline=bertscore_rescale_with_baseline,
    )
    return metrics, predictions, references
