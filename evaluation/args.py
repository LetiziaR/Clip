import argparse
import sys


def _flag_was_provided(flag):
    for token in sys.argv[1:]:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CoCa checkpoint (test loss + generation)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--language_model_path", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--decoder_model_path", type=str, default=None)
    parser.add_argument("--decoder_tokenizer_path", type=str, default=None)
    parser.add_argument("--dual_tokenizer", dest="dual_tokenizer", action="store_true")
    parser.add_argument("--no_dual_tokenizer", dest="dual_tokenizer", action="store_false")
    parser.set_defaults(dual_tokenizer=True)
    parser.add_argument("--ts_model_path", type=str, default="ts2vec_pretrained.pt")
    parser.add_argument("--patchtst_pretrained_name", type=str, default=None)
    parser.add_argument("--ts_arch", type=str, default="ts2vec", choices=["ts2vec", "patchtst"])
    parser.add_argument("--language_arch", type=str, default="bioclinicalbert", choices=["bert", "bioclinicalbert"])
    parser.add_argument("--decoder_arch", type=str, default="bart", choices=["bart", "gpt2", "t5", "biogpt"])
    parser.add_argument("--head_arch", type=str, default="mlp")
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--caption_loss_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0)
    parser.add_argument("--aux_classification_loss_weight", type=float, default=0.0)
    parser.add_argument("--enable_grouped_aux_heads", action="store_true")
    parser.add_argument("--disable_grouped_aux_heads", dest="enable_grouped_aux_heads", action="store_false")
    parser.set_defaults(enable_grouped_aux_heads=False)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500])
    parser.add_argument("--text_max_length", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="ptbxl", choices=["ptbxl", "mimic"])
    parser.add_argument("--text_source", type=str, default="report",
                        choices=["report", "pseudo_report", "note"])
    parser.add_argument("--return_labels", action="store_true")
    parser.add_argument("--label_col", type=str, default="scp_codes")
    parser.add_argument("--label_threshold", type=float, default=0.0)
    parser.add_argument("--normalize_mode", type=str, default="global",
                        choices=["global"])
    parser.add_argument("--mimic_notes_root", type=str, default=None)
    parser.add_argument("--mimic_demographics_dir", type=str, default=None)
    parser.add_argument("--mimic_folds_file", type=str, default="mimic_folds.csv",
                        help="CSV with subject_id,strat_fold columns")
    parser.add_argument("--mimic_max_samples", type=int, default=None)

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--skip_test_loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gen_max_new_tokens", type=int, default=96)
    parser.add_argument("--gen_num_beams", type=int, default=4)
    parser.add_argument("--gen_do_sample", dest="gen_do_sample", action="store_true")
    parser.add_argument("--gen_no_sample", dest="gen_do_sample", action="store_false")
    parser.set_defaults(gen_do_sample=False)
    parser.add_argument("--gen_temperature", type=float, default=0.8)
    parser.add_argument("--gen_top_p", type=float, default=0.95)
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--gen_repetition_penalty", type=float, default=1.15)
    parser.add_argument("--gen_length_penalty", type=float, default=1.0)
    parser.add_argument("--gen_max_batches", type=int, default=0)

    parser.add_argument("--compute_bertscore", dest="compute_bertscore", action="store_true")
    parser.add_argument("--no_compute_bertscore", dest="compute_bertscore", action="store_false")
    parser.set_defaults(compute_bertscore=True)
    parser.add_argument("--full_metrics", dest="full_metrics", action="store_true")
    parser.add_argument("--no_full_metrics", dest="full_metrics", action="store_false")
    parser.set_defaults(full_metrics=False)
    parser.add_argument("--bertscore_model_type", type=str, default="xlm-roberta-large")
    parser.add_argument(
        "--bertscore_model_alias",
        type=str,
        default=None,
        choices=["general_roberta", "general_xlm", "clinicalbert", "biobert", "pubmedbert"],
    )
    parser.add_argument("--bertscore_batch_size", type=int, default=16)
    parser.add_argument("--bertscore_lang", type=str, default="en")
    parser.add_argument("--bertscore_rescale_with_baseline", action="store_true")
    parser.add_argument("--no_bertscore_rescale_with_baseline", dest="bertscore_rescale_with_baseline", action="store_false")
    parser.set_defaults(bertscore_rescale_with_baseline=True)
    args = parser.parse_args()

    # Keep defaults aggressive for SR100, but safer for SR500 unless user overrides.
    if args.sampling_rate == 500:
        if not _flag_was_provided("--batch_size"):
            args.batch_size = 8
        if not _flag_was_provided("--gen_num_beams"):
            args.gen_num_beams = 1
        if not _flag_was_provided("--gen_max_new_tokens"):
            args.gen_max_new_tokens = 64

    return args
